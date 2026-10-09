# Training Mola

`train.py` trains Mola on an aligned Fiber-seq BAM with m6A and CpG (5mC) calls in the MM/ML tags.
In addition to the packages in `requirements.txt`, training requires:

```bash
pip install pysam pyBigWig
```

## Data

The released model was trained on the deep GM12878 Fiber-seq data of Vollger et al. (2025),
BioProject [PRJNA1233341](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1233341): reads aligned to
GRCh38 with pbmm2 v1.10.0 (CCS preset), with m6A called by fibertools-rs v0.1.4 and CpG
methylation by Primrose v1.4.0. The processed FIRE data are at
[doi:10.5281/zenodo.14511246](https://doi.org/10.5281/zenodo.14511246).

Windows are not drawn uniformly over the genome: their centers are sampled in proportion to the
per-base track supplied with `--weights`. We used the mean accessibility across molecules (the
percent-accessible track of the GM12878 FIRE data), so accessible regions are sampled more often.
The track is included in the Zenodo record with the model
([10.5281/zenodo.23108712](https://doi.org/10.5281/zenodo.23108712),
`gm12878_percent_accessible.bw`). The track can be a bigWig or a `.pth` file containing a
dictionary of per-chromosome arrays; without `--weights`, windows are drawn uniformly.

## One training step

1. Draw two 10 kb windows (`--windows`) and keep those fully spanned by 5 to 75 reads
   (`--min-reads`, `--max-reads`). A window outside that range is skipped, not subsampled, and a
   step with no window left is drawn again. Windows on the held-out chromosomes (chr8, chr9, and
   chr10) and windows that overlap one of the 38 regions in `hg38_blacklist.bed` are never used.
2. Turn each read into a target of shape (2, 10000): the m6A likelihood at A/T bases and the CpG
   methylation likelihood on both bases of each CpG, divided by 256. Calls below 125 out of 255
   count as 0.
3. Reverse-complement the sequence and reverse the targets with probability 0.5.
4. Give each read its own noise level and average the denoising loss over reads. By default
   (`--loss all`), the squared error is averaged over every position of both channels, as for the
   released model. `--loss masked` averages it only over the positions Fiber-seq measures: A/T
   bases for m6A and the two bases of each CpG for methylation. The model is therefore not trained
   to predict the zeros everywhere else. The two losses are on different scales, since the masked
   one averages over about a third of the positions.

## Running

```bash
python train/train.py --bam GM12878.fire.bam --genome resources/hg38.fa \
    --weights gm12878_percent_accessible.bw --name mola_gm12878
```

At intervals set by `--save-every` (default: 100 steps), the script writes
`models/<name>_model.pth`, a plain state dictionary that `mola.load_model` and
`python mola_predict.py sample <region> --model <checkpoint>` can read directly. It also writes
`models/<name>_optimizer.pth`, containing the optimizer state and step number. The training loss
is recorded in `models/<name>.log`.

- `--resume` continues from these checkpoints. `--steps` sets the total number of optimizer steps,
  including those completed before resuming.
- `--evaluate CHECKPOINT` does not train; it reports the validation numbers below for a checkpoint.
- `--loss masked` trains on the measured positions only (above). A model trained this way is not
  constrained elsewhere, so generate its molecules with
  `python mola_predict.py sample <region> --masked` or `mola.sample(..., masked=True)`. These set
  the unmeasured positions to 0 in the denoiser output. The validation loss and generated molecules
  use the same mask.
- `--reads-per-pass` sets how many reads go through the model at once. A batch can hold 150 reads,
  so the script runs it in parts and adds up the gradients. The default of 32 used 47 GB of GPU
  memory in our test; lower it on smaller GPUs. With several GPUs, the model runs with
  `nn.DataParallel`.

With bigWig weights for the whole genome, the script uses about 35 GB of system memory, and a
step takes about 6 s on one H200.

## Options

`python train/train.py -h` lists the same options with their defaults.

| Option | Default | Description |
|---|---|---|
| **Data** | | |
| `--bam` | (required) | Indexed, aligned Fiber-seq BAM with m6A and 5mC calls in the MM/ML tags |
| `--genome` | `resources/hg38.fa` | Reference FASTA, indexed on first use |
| `--weights` | uniform | Per-base track that window centers are drawn in proportion to: a bigWig, or a `.pth` dict of per-chromosome arrays |
| `--blacklist` | `train/hg38_blacklist.bed` | BED of regions; windows that overlap them are skipped (`''` for none) |
| `--holdout` | `chr8,chr9,chr10` | Chromosomes never used for training |
| `--length` | 10000 | Window length in bp; a multiple of 400, since the U-Net pools by 4 × 4 × 5 × 5 |
| `--min-reads` | 5 | Skip a window if fewer reads span all of it |
| `--max-reads` | 75 | Skip a window if more reads span all of it; also caps the reads in the validation loss |
| **Training** | | |
| `--steps` | 1000000 | Total optimizer steps, counted across `--resume` |
| `--windows` | 2 | Windows drawn per step, before the read-count filter |
| `--lr` | 0.001 | Adam learning rate |
| `--loss` | `all` | `all`: loss over every position of both channels; `masked`: only over A/T bases for m6A and CpGs for methylation |
| `--reads-per-pass` | 32 | Reads per forward/backward pass; gradients are added up over passes, so this changes memory use, not the result |
| `--seed` | 5 | Random seed for window draws, noise, and initialization |
| **Checkpoints** | | |
| `--outdir` | `models` | Directory for the checkpoints, the log, and the validation table |
| `--name` | `mola` | Prefix of the files in `--outdir` |
| `--save-every` | 100 | Steps between checkpoints and training-loss lines in the log |
| `--resume` | off | Continue from the checkpoints of `--name` |
| **Validation** | | |
| `--validation` | `chr10` | Chromosomes the validation windows come from; they must be in `--holdout` |
| `--validate-every` | 1000 | Steps between validations; 0 turns validation off |
| `--val-windows` | 4 | Validation windows, drawn once like the training windows, each with at least 20 spanning reads |
| `--val-molecules` | 100 | Molecules generated per validation window |
| `--evaluate` | off | Do not train; report the validation loss and correlations of a checkpoint |

## Validation

At intervals set by `--validate-every` (default: 1,000 steps; 0 disables validation), the script
checks the model on a fixed set of `--val-windows` windows (default: 4) from the `--validation`
chromosomes (default: chr10, held out from training). The windows are drawn once, with the same
weighting as the training windows, and each has at least 20 spanning reads. For each window, it
computes:

- the loss on the measured reads, with seeded noise so that steps can be compared;
- the agreement between `--val-molecules` (default: 100) generated molecules and the measured
  reads (`mola.agreement`): the Pearson r between mean accessibility profiles at A/T bases and the
  mean Pearson r between corresponding rows of co-accessibility maps (100 bp bins of A/T bases over
  the 10 kb window).

The averages across validation windows are written to the log and to
`models/<name>.validation.tsv`. `--evaluate CHECKPOINT` reports the same three numbers for a saved
model without training.

A validation run takes about 40 s on one H200. Its random draws are kept separate from those used
for training, so training draws the same windows and noise with or without validation. With only
a few windows and 100 molecules, the correlations are noisier than an evaluation with 500
molecules per DHS across the held-out chromosomes, but they show whether a run is improving.
