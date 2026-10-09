"""Pseudo-dynamic trajectories with fixed-sigma Langevin sampling.

Usage:
    python dynamics/dynamics.py <region> [-n 10] [--trajectory-steps 10000] [-o out.npz]
    python mola_predict.py summarize out.npz [--frame -1]

Regions are hg38, 0-based and end-exclusive, e.g. chr19:47509000-47519000 (10 kb). Run
`python dynamics/dynamics.py -h` for the full list of options.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mola
from mola_predict import add_common, get_sequence


def run_dynamics(args):
    chrom, start, end = mola.parse_region(args.region)
    if (end - start) % 400 != 0:
        raise SystemExit("region length must be a multiple of 400 bp; Mola was trained on 10 kb")
    sequence = get_sequence(mola.load_genome(args.genome), chrom, start, end)
    model = mola.load_model(args.model, args.device)

    torch.manual_seed(args.seed)
    x0 = mola.sample(model, sequence, args.num, steps=args.steps, sampler=args.sampler, batch_size=args.batch_size,
                     device=args.device, masked=args.masked)
    frames = mola.langevin(model, x0, sequence, args.trajectory_steps, sigma=args.sigma, step_size=args.step_size,
                           clip=args.clip, noise=args.noise, burn_in=args.burn_in, readout_steps=args.readout_steps,
                           save_every=args.save_every, device=args.device, masked=args.masked)
    np.savez(args.output, signals=frames, sequence=sequence.astype(np.float32), chrom=chrom, start=start, end=end,
             sigma=args.sigma, step_size=args.step_size, noise=args.noise, clip=args.clip, burn_in=args.burn_in,
             save_every=args.save_every, readout_steps=args.readout_steps, sampler=args.sampler, steps=args.steps)
    print(f"saved trajectories with shape {frames.shape} to {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Mola pseudo-dynamic trajectories with fixed-sigma Langevin sampling.")
    p.add_argument("region", help="chrom:start-end, e.g. chr19:47509000-47519000")
    p.add_argument("-o", "--output", default="mola_dynamics.npz")
    p.add_argument("--burn-in", type=int, default=5000, help="Langevin updates before the first saved frame (default 5000)")
    p.add_argument("--trajectory-steps", type=int, default=10000, help="Langevin updates after the burn-in (default 10000)")
    p.add_argument("--save-every", type=int, default=1, help="save every n-th frame, starting with the first post-burn-in update (default 1)")
    p.add_argument("--sigma", type=float, default=0.9,
                   help="fixed noise level at which the score is evaluated, and the standard deviation of the "
                        "initial perturbation (default 0.9)")
    p.add_argument("--step-size", type=float, default=0.01, help="Langevin step size (default 0.01)")
    p.add_argument("--clip", type=float, default=10.0,
                   help="largest root-mean-square of the score per molecule; 0 turns clipping off (default 10)")
    p.add_argument("--noise", type=float, default=1.0,
                   help="multiplier on the Gaussian noise added at each update: larger values explore more, "
                        "0 leaves only the score-directed drift (default 1)")
    p.add_argument("--readout-steps", type=int, default=1,
                   help="denoiser steps for the clean-state readout of each saved frame: 1 is one denoiser call at "
                        "sigma; 0 saves the noisy state (default 1)")
    add_common(p, 10)
    run_dynamics(p.parse_args())
