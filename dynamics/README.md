# Pseudo-dynamics with Mola

`dynamics.py` simulates pseudo-dynamic trajectories with sequence-conditioned Langevin sampling.
Each trajectory starts from a generated molecule perturbed with Gaussian noise at a fixed noise
level σ. Each update then combines a score-directed drift toward more probable chromatin states
with independent Gaussian noise:

```text
x ← x + η · s(x) + noise · √(2η) · ξ,    ξ ~ N(0, I)
```

where s(x) is the model score at σ, rescaled when its root-mean-square exceeds the clip. The drift
favors locally higher-density configurations, while the noise lets the chain explore alternative
states. After a burn-in, a frame is saved after each update: the clean-state readout from one
denoiser call at σ. The readout does not replace the noisy state used for the next update.

## Running

Run the script from the repository root, with the trained model and genome in `resources/` (see
the [main README](../README.md#installation)):

```bash
python dynamics/dynamics.py chr19:47509000-47519000 -n 10 -o napa_dynamics.npz
python mola_predict.py summarize napa_dynamics.npz --frame -1
```

`summarize` writes the same tables and figure as for `sample`, for one frame of the trajectories
(`--frame`, by default the last).

With the default settings, 10 molecules take about 3 hours on one A100. The output has shape
(molecules, 2, length, frames) in float16, so 10,000 frames of 10 molecules take 4 GB;
`--save-every` thins the frames, and `--burn-in` and `--trajectory-steps` shorten the run. The
frames are not clipped. `summarize --bam --frame <i>` writes one frame as a BAM for calling
footprints with FiberHMM (see the [main README](../README.md#calling-footprints-with-fiberhmm)).
The trajectories are ordered samples in model pseudo-time: they show which states are
stable and how molecules may move between them, not calibrated biological time.

## Key hyperparameters

All of them can be changed:

| Option | Default | Description |
|---|---|---|
| `--noise` | 1.0 | Multiplier on the Gaussian noise added at each update. Larger values explore more and leave states sooner; 0 leaves only the score-directed drift |
| `--sigma` | 0.9 | Fixed noise level σ at which the score is evaluated, and the standard deviation of the initial perturbation. A higher σ uses a smoother version of the learned distribution |
| `--step-size` | 0.01 | Langevin step size η |
| `--burn-in`, `--trajectory-steps` | 5000, 10000 | Updates before the first saved frame, and updates after it |
| `-n` | 10 | Number of trajectories |

`--clip`, `--save-every`, and `--readout-steps` are described by `python dynamics/dynamics.py -h`.
The starting molecules are generated as by `mola_predict.py sample`, with the same `--sampler`,
`--steps`, `--seed`, `--batch-size`, `--device`, `--masked`, `--model`, and `--genome` options.

## Python

To start trajectories from generated molecules:

```python
import torch
import mola

device = "cuda"
model = mola.load_model("resources/mola.pth", device=device)
genome = mola.load_genome("resources/hg38.fa")
seq = genome.get_encoding_from_coords("chr19", 47509000, 47519000)

torch.manual_seed(0)
x = mola.sample(model, seq, 4, device=device)
frames = mola.langevin(model, x, seq, steps=200, burn_in=100, device=device)  # (4, 2, 10000, 200), float16
```

To write every frame to one BAM for FiberHMM, with read names that keep the molecule and frame:

```python
x = frames.transpose(0, 3, 1, 2).reshape(-1, 2, frames.shape[2])        # (molecules × frames, 2, length)
names = [f"mola_{i}_frame{t}" for i in range(frames.shape[0]) for t in range(frames.shape[3])]
mola.write_bam(x, seq, "chr19", 47509000, "napa_frames.bam", names)
```

`mola.langevin` has the same defaults as `dynamics.py` (`sigma=0.9`, `step_size=0.01`,
`noise=1.0`, `clip=10`, `burn_in=5000`), except that `steps` must be given; the script uses
10,000. Each can be changed, for example `noise=0.5` for a chain that explores less.
