"""Train Mola on a Fiber-seq BAM. See train/README.md.

    python train/train.py --bam GM12878.fire.bam --weights gm12878_percent_accessible.bw --name mola_gm12878
"""
import argparse
import logging
import os
import sys

import numpy as np
import pysam
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mola import Mola, agreement, load_genome, load_weights, measured_mask, sample, training_loss


class FiberData:
    """m6A and CpG likelihoods of the reads that fully span a window, as (reads, 2, length).

    Calls with a likelihood (ML) below 125 are dropped, and alignments are not filtered by flag.
    """

    def __init__(self, bam):
        self.path = bam
        self.bam = None

    def get(self, chrom, start, end):
        if self.bam is None:  # open lazily so the object can be passed to worker processes
            self.bam = pysam.AlignmentFile(self.path)
        reads = []
        for read in self.bam.fetch(chrom, start, end):
            if read.is_unmapped or read.reference_start > start or read.reference_end < end:
                continue
            length = read.reference_end - read.reference_start
            ref = np.array([-1 if p is None else p for p in read.get_reference_positions(full_length=True)])
            x = np.zeros((2, length), dtype=np.float32)
            for (base, strand, mod), calls in (read.modified_bases or {}).items():
                calls = np.array(calls, dtype=np.int64).reshape(-1, 2)
                pos = ref[calls[:, 0]]
                keep = (pos >= 0) & (calls[:, 1] >= 125)
                pos, ml = pos[keep] - read.reference_start, calls[keep, 1]
                if mod == "a":
                    x[0, pos] = ml
                elif mod == "m":
                    # a CpG call is on one base of the CpG; copy it to the other base too
                    x[1, pos] = ml
                    other = pos - 1 if read.is_reverse else pos + 1
                    inside = (other >= 0) & (other < length)
                    x[1, other[inside]] = ml[inside]
            reads.append(x[:, start - read.reference_start:end - read.reference_start])
        return np.array(reads, dtype=np.float32).reshape(-1, 2, end - start)


class WindowSampler:
    """Draw windows with centers sampled in proportion to a per-base weight track."""

    def __init__(self, genome, weights, length, holdout):
        self.length = length
        self.chroms, self.cdfs, totals = [], [], []
        for chrom, size in genome.get_chr_lens():
            if chrom in holdout or chrom not in weights:
                continue
            w = np.asarray(weights[chrom], dtype=np.float64).copy()
            w[:length // 2] = 0
            w[size - length // 2:] = 0
            if w.sum() > 0:
                self.chroms.append(chrom)
                self.cdfs.append(np.cumsum(w / w.sum()))
                totals.append(w.sum())
        self.p = np.array(totals) / np.sum(totals)

    def sample(self):
        i = np.random.choice(len(self.chroms), p=self.p)
        center = int(np.searchsorted(self.cdfs[i], np.random.rand(), "right"))
        return self.chroms[i], center - self.length // 2, center + self.length // 2


def read_weights(path, genome):
    """Per-base weights for each chromosome from a bigWig or from a torch file (dict or list in genome order)."""
    if path is None:
        return {chrom: np.ones(size, dtype=np.float32) for chrom, size in genome.get_chr_lens()}
    if path.endswith((".bw", ".bigWig", ".bigwig")):
        import pyBigWig
        bw = pyBigWig.open(path)
        weights = {}
        for chrom, size in genome.get_chr_lens():
            if chrom in bw.chroms():
                weights[chrom] = np.nan_to_num(bw.values(chrom, 0, min(size, bw.chroms(chrom)), numpy=True)).astype(np.float32)
                weights[chrom].resize(size, refcheck=False)
        return weights
    weights = torch.load(path, weights_only=False)
    if isinstance(weights, dict):
        return weights
    return {chrom: w for (chrom, _), w in zip(genome.get_chr_lens(), weights)}


def batch_loss(model, seq, target, chunk, backward=False, masked=False):
    """Mean training loss of a batch, run through the model `chunk` reads at a time. With backward,
    the gradients of the chunks add up to the gradient of the whole batch (GroupNorm works per
    read), so a batch of 150 reads fits on one GPU. masked restricts the loss to the positions
    Fiber-seq measures (see mola.training_loss)."""
    total = 0.0
    for i in range(0, len(target), chunk):
        s, t = seq[i:i + chunk], target[i:i + chunk]
        loss = training_loss(model, t, s, measured_mask(s) if masked else None) * len(t) / len(target)
        if backward:
            loss.backward()
        total += loss.item()
    return total


def validation_windows(genome, data, sampler, n, min_reads=20, seed=0):
    """A fixed set of n held-out windows with at least `min_reads` spanning reads, as (name, sequence, reads)."""
    state = np.random.get_state()  # keep the training draws the same with or without validation
    np.random.seed(seed)
    windows, tries = [], 0
    while len(windows) < n:
        tries += 1
        if tries > 100000:
            raise SystemExit(f"found only {len(windows)} validation windows with at least {min_reads} reads")
        chrom, start, end = sampler.sample()
        seq = genome.get_encoding_from_coords(chrom, start, end)
        reads = data.get(chrom, start, end)
        if len(seq) == end - start and len(reads) >= min_reads:
            windows.append((f"{chrom}:{start}-{end}", seq, reads / 256))
    np.random.set_state(state)
    return windows


def validate(net, windows, n_molecules, chunk, max_reads, device, masked=False):
    """Loss on the validation windows and the agreement of generated with observed molecules
    (mola.agreement): the Pearson r of mean accessibility at each A/T base, and of co-accessibility
    (100 bp bins of A/T bases, maps compared row by row). Averaged over windows.
    The noise is seeded, so the numbers are comparable between steps. masked uses the masked loss and
    generates masked molecules, as for a model trained that way."""
    net.eval()
    losses, r_mean, r_coacc = [], [], []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == "cuda" else []), torch.no_grad():
        torch.manual_seed(0)
        for _, seq, observed in windows:
            target = torch.as_tensor(observed[:max_reads], device=device)
            s = torch.as_tensor(seq.T[None], dtype=torch.float32, device=device).expand(len(target), -1, -1)
            losses.append(batch_loss(net, s, target, chunk, masked=masked))
            a = agreement(sample(net, seq, n_molecules, device=device, masked=masked), observed, seq)
            r_mean.append(a["r_mean"])
            r_coacc.append(a["r_coacc"])
    net.train()
    return np.mean(losses), np.mean(r_mean), np.mean(r_coacc)


def get_batch(genome, data, sampler, windows, min_reads, max_reads, device):
    """Draw `windows` windows and keep those with min_reads..max_reads spanning reads (None if none is kept)."""
    seqs, targets = [], []
    for _ in range(windows):
        chrom, start, end = sampler.sample()
        target = data.get(chrom, start, end)
        if not min_reads <= len(target) <= max_reads:
            continue
        seq = genome.get_encoding_from_coords(chrom, start, end)
        if len(seq) != end - start:  # overlaps the blacklist
            continue
        seqs.append(torch.as_tensor(seq.T[None], dtype=torch.float32).expand(len(target), -1, -1))
        targets.append(torch.as_tensor(target) / 256)
    if not seqs:
        return None
    return torch.cat(seqs).to(device), torch.cat(targets).to(device)


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    # show a default only where there is one
    def _get_help_string(self, action):
        if action.required or action.default in (None, False):
            return action.help
        return super()._get_help_string(action)


parser = argparse.ArgumentParser(description="Train Mola on a Fiber-seq BAM. See train/README.md.",
                                 formatter_class=HelpFormatter)
data_args = parser.add_argument_group("data")
data_args.add_argument("--bam", required=True, help="indexed, aligned Fiber-seq BAM with m6A and 5mC calls in the MM/ML tags")
data_args.add_argument("--genome", default="resources/hg38.fa", help="reference FASTA, indexed on first use")
data_args.add_argument("--weights", default=None,
                       help="per-base track that window centers are drawn in proportion to: a bigWig, or a .pth dict of "
                            "per-chromosome arrays; windows are drawn uniformly if not given")
data_args.add_argument("--blacklist", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "hg38_blacklist.bed"),
                       help="BED of regions; windows that overlap them are skipped ('' for none)")
data_args.add_argument("--holdout", default="chr8,chr9,chr10", help="comma-separated chromosomes never used for training")
data_args.add_argument("--length", type=int, default=10000,
                       help="window length in bp; a multiple of 400, since the U-Net pools by 4 x 4 x 5 x 5")
data_args.add_argument("--min-reads", type=int, default=5,
                       help="skip a window if fewer reads span all of it")
data_args.add_argument("--max-reads", type=int, default=75,
                       help="skip a window if more reads span all of it (it is not subsampled); also caps the reads used "
                            "for the validation loss")

train_args = parser.add_argument_group("training")
train_args.add_argument("--steps", type=int, default=1000000, help="total optimizer steps, counted across --resume")
train_args.add_argument("--windows", type=int, default=2,
                        help="windows drawn per step; those outside --min-reads..--max-reads are dropped, and a step "
                             "with none left is drawn again")
train_args.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
train_args.add_argument("--loss", choices=["all", "masked"], default="all",
                        help="all: average the loss over every position of both channels, as for the released model; "
                             "masked: only over A/T bases for m6A and CpGs for methylation, the positions Fiber-seq measures")
train_args.add_argument("--reads-per-pass", type=int, default=32,
                        help="reads per forward/backward pass; the gradients of the passes are added up, so this trades "
                             "GPU memory for speed only (32 used 47 GB)")
train_args.add_argument("--seed", type=int, default=5, help="random seed for window draws, noise and initialization")

save_args = parser.add_argument_group("checkpoints")
save_args.add_argument("--outdir", default="models", help="directory for checkpoints, the log and the validation table")
save_args.add_argument("--name", default="mola", help="prefix of the files written to --outdir")
save_args.add_argument("--save-every", type=int, default=100, help="steps between checkpoints and training-loss lines in the log")
save_args.add_argument("--resume", action="store_true", help="continue from the checkpoints of --name in --outdir")

val_args = parser.add_argument_group("validation")
val_args.add_argument("--validation", default="chr10",
                      help="comma-separated chromosomes the validation windows come from; they must be in --holdout")
val_args.add_argument("--validate-every", type=int, default=1000, help="steps between validations; 0 turns validation off")
val_args.add_argument("--val-windows", type=int, default=4,
                      help="number of validation windows, drawn once like the training windows, each with at least 20 "
                           "spanning reads")
val_args.add_argument("--val-molecules", type=int, default=100, help="molecules generated per validation window")
val_args.add_argument("--evaluate", default=None, metavar="CHECKPOINT",
                      help="do not train; report the validation loss and correlations of CHECKPOINT")
args = parser.parse_args()
holdout, validation = args.holdout.split(","), args.validation.split(",")
masked = args.loss == "masked"
if not set(validation) <= set(holdout):
    raise SystemExit("the --validation chromosomes have to be in --holdout, so that they are not trained on")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.outdir, exist_ok=True)
logging.basicConfig(filename=f"{args.outdir}/{args.name}.log", level=logging.INFO, format="%(asctime)s %(message)s")
logging.info(f"loss: {args.loss}")

genome = load_genome(args.genome, blacklist=args.blacklist or None)
data = FiberData(args.bam)
weights = read_weights(args.weights, genome)
sampler = WindowSampler(genome, weights, args.length, holdout)
val_sampler = WindowSampler(genome, {c: w for c, w in weights.items() if c in validation}, args.length, holdout=[])
del weights
val_windows = []
if args.validate_every or args.evaluate:
    val_windows = validation_windows(genome, data, val_sampler, args.val_windows)
    logging.info("validation windows: " + ", ".join(name for name, _, _ in val_windows))

net = Mola().to(device)
model = torch.nn.DataParallel(net) if torch.cuda.device_count() > 1 else net
model_path = f"{args.outdir}/{args.name}_model.pth"
optimizer_path = f"{args.outdir}/{args.name}_optimizer.pth"

if args.evaluate:
    net.load_state_dict(load_weights(args.evaluate, device))
    loss, r_mean, r_coacc = validate(net, val_windows, args.val_molecules, args.reads_per_pass, args.max_reads, device, masked)
    print(f"{args.evaluate} on {len(val_windows)} {args.validation} windows: validation loss {loss:.5f} ({args.loss}) | "
          f"r mean accessibility {r_mean:.3f} | r co-accessibility {r_coacc:.3f}")
    raise SystemExit

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
step = 0
if args.resume and os.path.exists(model_path):
    net.load_state_dict(load_weights(model_path, device))
    state = torch.load(optimizer_path, map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    step = state["step"] + 1
    print(f"resuming from step {step}")

model.train()
losses = []
while step < args.steps:
    batch = get_batch(genome, data, sampler, args.windows, args.min_reads, args.max_reads, device)
    if batch is None:
        continue
    seq, target = batch
    if torch.rand(1) < 0.5:  # reverse complement
        seq, target = seq.flip([1, 2]), target.flip([2])

    optimizer.zero_grad()
    losses.append(batch_loss(model, seq, target, args.reads_per_pass, backward=True, masked=masked))
    optimizer.step()

    if step % args.save_every == 0 or step == args.steps - 1:
        msg = f"{step} | loss {np.mean(losses):.5f} | {len(target)} reads"
        print(msg, flush=True)
        logging.info(msg)
        losses = []
        torch.save(net.state_dict(), model_path)
        torch.save({"step": step, "optimizer": optimizer.state_dict()}, optimizer_path)

    if args.validate_every and (step % args.validate_every == 0 or step == args.steps - 1):
        loss, r_mean, r_coacc = validate(net, val_windows, args.val_molecules, args.reads_per_pass, args.max_reads, device, masked)
        msg = f"{step} | validation loss {loss:.5f} | r mean accessibility {r_mean:.3f} | r co-accessibility {r_coacc:.3f}"
        print(msg, flush=True)
        logging.info(msg)
        table = f"{args.outdir}/{args.name}.validation.tsv"
        if not os.path.exists(table):
            open(table, "w").write("step\tloss\tr_mean_accessibility\tr_coaccessibility\n")
        open(table, "a").write(f"{step}\t{loss:.6f}\t{r_mean:.4f}\t{r_coacc:.4f}\n")
    step += 1
