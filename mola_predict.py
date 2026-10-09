"""Mola command-line interface.

Usage:
    python mola_predict.py sample    <region> [-n 100] [-o out.npz]
    python mola_predict.py variant   <chrom:pos> <ref> <alt> [-n 100] [-o out.npz]
    python mola_predict.py summarize <file.npz> [--frame -1]

Regions are hg38, 0-based and end-exclusive, e.g. chr19:47509000-47519000 (10 kb).
Variant positions are 1-based, as in a VCF. Run `python mola_predict.py <command> -h`
for the full list of options for each command. Pseudo-dynamic trajectories are simulated
with dynamics/dynamics.py, and summarize reads its output too.
"""
import argparse
import os

import numpy as np
import torch

import mola


def add_common(p, n):
    p.add_argument("-n", "--num", type=int, default=n, help=f"number of molecules (default {n})")
    p.add_argument("--model", default="resources/mola.pth", help="trained model (default resources/mola.pth)")
    p.add_argument("--genome", default="resources/hg38.fa", help="reference FASTA (default resources/hg38.fa)")
    p.add_argument("--sampler", choices=["euler", "heun"], default="euler",
                   help="ODE integrator for generating molecules: euler, or heun (second order, about twice the "
                        "network evaluations per step) (default euler)")
    p.add_argument("--steps", type=int, default=10,
                   help="steps of the noise schedule for generating molecules (default 10)")
    p.add_argument("--batch-size", type=int, default=50, help="molecules per model call; lower it if GPU memory runs out (default 50)")
    p.add_argument("--seed", type=int, default=0, help="random seed; the same seed gives the same molecules (default 0)")
    p.add_argument("--device", default="cuda", help="cuda or cpu (default cuda)")
    p.add_argument("--masked", action="store_true",
                   help="for a model trained with --loss masked: m6A only at A/T bases and methylation only at CpGs")


def get_sequence(genome, chrom, start, end):
    sequence = genome.get_encoding_from_coords(chrom, start, end)
    if len(sequence) != end - start:
        raise SystemExit(f"{chrom}:{start}-{end} is not inside a chromosome of the genome FASTA "
                         f"(names look like {genome.get_chr_lens()[0][0]})")
    return sequence


def run_sample(args):
    chrom, start, end = mola.parse_region(args.region)
    if (end - start) % 400 != 0:
        raise SystemExit("region length must be a multiple of 400 bp; Mola was trained on 10 kb")
    sequence = get_sequence(mola.load_genome(args.genome), chrom, start, end)
    model = mola.load_model(args.model, args.device)

    torch.manual_seed(args.seed)
    x = mola.sample(model, sequence, args.num, steps=args.steps, sampler=args.sampler, batch_size=args.batch_size,
                    device=args.device, masked=args.masked)
    if args.binarize:
        x = np.round(x).clip(0, 1)
    np.savez(args.output, signals=x.astype(np.float32), sequence=sequence.astype(np.float32),
             chrom=chrom, start=start, end=end, sampler=args.sampler, steps=args.steps)
    print(f"saved {len(x)} molecules for {chrom}:{start}-{end} to {args.output}")


def run_variant(args):
    chrom, pos = args.variant.split(":")
    pos = int(pos.replace(",", "")) - 1
    ref, alt = args.ref.upper(), args.alt.upper()
    if len(ref) != len(alt):
        raise SystemExit("only substitutions are supported (REF and ALT must have the same length)")

    start = pos - args.window // 2
    end = start + args.window
    offset = pos - start
    genome = mola.load_genome(args.genome)
    seq_ref = get_sequence(genome, chrom, start, end)
    observed = genome.get_sequence_from_coords(chrom, pos, pos + len(ref)).upper()
    if observed != ref:
        raise SystemExit(f"REF allele {ref} does not match the genome ({observed}) at {chrom}:{pos + 1}")
    seq_alt = mola.insert(seq_ref, alt, offset)

    # both alleles use the same seed, so the difference between them is not sampling noise alone
    model = mola.load_model(args.model, args.device)
    torch.manual_seed(args.seed)
    x_ref = mola.sample(model, seq_ref, args.num, steps=args.steps, sampler=args.sampler, batch_size=args.batch_size,
                        device=args.device, masked=args.masked)
    torch.manual_seed(args.seed)
    x_alt = mola.sample(model, seq_alt, args.num, steps=args.steps, sampler=args.sampler, batch_size=args.batch_size,
                        device=args.device, masked=args.masked)

    # the mean-accessibility effect: log2 ratio of mean m6A over the central 200 bp
    mu_ref = x_ref[:, 0, offset - 100:offset + 100].mean()
    mu_alt = x_alt[:, 0, offset - 100:offset + 100].mean()
    print(f"{chrom}:{pos + 1} {ref}>{alt}  mean m6A REF {mu_ref:.4f}  ALT {mu_alt:.4f}  "
          f"log2FC {np.log2((mu_alt + 1e-6) / (mu_ref + 1e-6)):.3f}")
    np.savez(args.output, ref=x_ref.astype(np.float32), alt=x_alt.astype(np.float32),
             sequence_ref=seq_ref.astype(np.float32), sequence_alt=seq_alt.astype(np.float32),
             chrom=chrom, start=start, end=end, pos=pos + 1, ref_allele=ref, alt_allele=alt, sampler=args.sampler,
             steps=args.steps)
    print(f"saved {args.num} molecules per allele to {args.output}")


def chrom_lengths(genome):
    """Chromosome lengths from the FASTA index, or nothing if there is no index."""
    if not os.path.exists(f"{genome}.fai"):
        return {}
    return {line.split("\t")[0]: int(line.split("\t")[1]) for line in open(f"{genome}.fai")}


def run_summarize(args):
    data = np.load(args.input)
    prefix = args.prefix or args.input.removesuffix(".npz")
    chrom = str(data["chrom"]) if "chrom" in data else ""
    figures = [f"{prefix}.{f}" for f in args.format]
    start = int(data["start"]) if "start" in data else 0
    if args.bam and not chrom:
        raise SystemExit(f"--bam needs the coordinates that sample, variant, and dynamics save; {args.input} has none")
    length = chrom_lengths(args.genome).get(chrom) if args.bam else None

    if args.key is None and "ref" in data and "alt" in data:
        # variant output: a summary for each allele and a figure comparing them
        for key in ("ref", "alt"):
            mola.write_summary(data[key].astype(np.float32), data[f"sequence_{key}"], start, f"{prefix}.{key}", args.bin, chrom,
                               args.format)
            if args.bam:
                mola.write_bam(data[key], data[f"sequence_{key}"], chrom, start, f"{prefix}.{key}.bam",
                               [f"{key}_{i}" for i in range(len(data[key]))], length)
        label = f"{data['ref_allele']}>{data['alt_allele']}"
        mola.plot_variant(data["ref"].astype(np.float32), data["alt"].astype(np.float32), start, data["sequence_ref"],
                          data["sequence_alt"], int(data["pos"]) - 1, label, chrom, figures, args.bin)
        print(f"wrote {prefix}.ref.*, {prefix}.alt.*, and {', '.join(figures)}"
              + (f", with {prefix}.ref.bam and {prefix}.alt.bam" if args.bam else ""))
        return

    key = args.key or "signals"
    if key not in data:
        raise SystemExit(f"{args.input} has no '{key}' array, pick one with --key: {', '.join(data.files)}")
    x = data[key]
    names = [f"mola_{i}" for i in range(len(x))]
    if x.ndim == 4:  # dynamics output, the last axis is time
        frame = args.frame % x.shape[-1]
        x = x[..., frame]
        names = [f"mola_{i}_frame{frame}" for i in range(len(x))]
    seq_key = f"sequence_{key}" if f"sequence_{key}" in data else "sequence"
    sequence = data[seq_key] if seq_key in data else None
    mola.write_summary(x.astype(np.float32), sequence, start, prefix, args.bin, chrom, args.format)
    outputs = f"{prefix}.mean.csv, {prefix}.molecules.csv, {prefix}.coaccessibility.npy, {', '.join(figures)}"
    if args.bam:
        mola.write_bam(x, sequence, chrom, start, f"{prefix}.bam", names, length)
        outputs += f", {prefix}.bam"
    print(f"wrote {outputs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mola: single-molecule chromatin states from DNA sequence.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sample", help="generate molecules for a genomic region")
    p.add_argument("region", help="chrom:start-end, e.g. chr19:47509000-47519000")
    p.add_argument("-o", "--output", default="mola_samples.npz")
    p.add_argument("--binarize", action="store_true", help="round outputs to 0/1 (by default the continuous values are kept)")
    add_common(p, 100)
    p.set_defaults(func=run_sample)

    p = sub.add_parser("variant", help="generate molecules for the REF and ALT alleles of a substitution")
    p.add_argument("variant", help="chrom:pos, 1-based, e.g. chr17:19447245")
    p.add_argument("ref", help="reference allele, e.g. C or CC")
    p.add_argument("alt", help="alternative allele, same length as REF")
    p.add_argument("-o", "--output", default="mola_variant.npz")
    p.add_argument("--window", type=int, default=10000)
    add_common(p, 100)
    p.set_defaults(func=run_variant)

    p = sub.add_parser("summarize", help="mean tracks, per-molecule means, co-accessibility, and a plot")
    p.add_argument("input", help=".npz written by sample, variant, or dynamics/dynamics.py")
    p.add_argument("--key", default=None, help="array to read (default: signals, or both alleles of a variant file)")
    p.add_argument("--frame", type=int, default=-1, help="frame of a dynamics trajectory (default: the last)")
    p.add_argument("--prefix", default=None, help="output prefix (default: the input name)")
    p.add_argument("--bin", type=int, default=20, help="bin size for co-accessibility (default 20 bp)")
    p.add_argument("--format", nargs="+", choices=["png", "pdf", "svg"], default=["png"],
                   help="figure formats; --format png pdf writes both, and PDF text stays editable (default png)")
    p.add_argument("--bam", action="store_true",
                   help="also write the molecules as aligned reads with m6A and CpG methylation in MM/ML tags, "
                        "for FiberHMM or other Fiber-seq tools (needs pysam)")
    p.add_argument("--genome", default="resources/hg38.fa",
                   help="FASTA whose index gives the chromosome length in the BAM header (default resources/hg38.fa)")
    p.set_defaults(func=run_summarize)

    args = parser.parse_args()
    args.func(args)
