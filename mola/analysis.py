"""Summaries and plots of a set of molecules.

`x` is always an array of molecules with shape (n, 2, length), channel 0 m6A and channel 1 CpG
methylation, as returned by `mola.sample`. The plots show an orange mean-accessibility track, purple m6A on
single molecules, and co-accessibility as a triangle in the cooltools "fall" colors (0 to 0.8).
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from .sequence import at_mask, cpg_mask, to_bases


# plot colors
INK = "#14161f"
MUTED = "#7a8093"
RULE = "#f1f1f2"
BORDER = "#eaeaeb"
SPAN = "#e8e8e9"
M6A = "#800080"
MODEL = "#f6a04d"
REF = "#5b6472"
DANGER = "#d64545"
fall = LinearSegmentedColormap.from_list("fall", ["#ffffff", "#ffffcd", "#ffeda0", "#fed977", "#feb24c", "#fd8c3c",
                                                  "#fc4e2a", "#e2191c", "#bd0026", "#7d0025", "#000000"])
diverging = LinearSegmentedColormap.from_list("mola_diverging", ["#2a78d6", "#eeeeee", "#e34948"])

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.monospace": ["Menlo", "Consolas", "Liberation Mono", "DejaVu Sans Mono"],
    "font.size": 10,
    "text.color": INK,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "pdf.fonttype": 42,
}
DPI = 300


# ---------------------------------------------------------------- numbers

def bin_mean(x, binsize):
    """Average the last axis in bins of `binsize`, ignoring NaN (a trailing partial bin is dropped)."""
    length = x.shape[-1] // binsize * binsize
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # bins without a value give NaN
        return np.nanmean(x[..., :length].reshape(*x.shape[:-1], -1, binsize), axis=-1)


def accessibility_track(x, sequence=None, binsize=20):
    """Mean m6A across molecules in bins. With the sequence, only A/T bases are averaged,
    since m6A cannot occur at C/G."""
    m = x[:, 0].mean(axis=0)
    if sequence is not None:
        m = np.where(at_mask(sequence), m, np.nan)
    return bin_mean(m, binsize)


def coaccessibility(m6a, binsize=20, sequence=None):
    """Pearson correlation of binned m6A between every pair of bins, across molecules.

    m6a has shape (n, length). With the sequence, each bin averages only its A/T bases, the
    positions where m6A can be measured. Bins that never vary across molecules give NaN.
    """
    m6a = np.asarray(m6a, dtype=float)
    if sequence is not None:
        m6a = np.where(at_mask(sequence), m6a, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.corrcoef(bin_mean(m6a, binsize), rowvar=False)


def _row_correlation(a, b, min_valid=0.3):
    # mean Pearson r between corresponding rows of two maps, over rows with > min_valid finite pairs
    r = []
    for x, y in zip(a, b):
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.mean() > min_valid and ok.sum() > 2:
            r.append(np.corrcoef(x[ok], y[ok])[0, 1])
    return np.nanmean(r) if r else np.nan


def agreement(generated, observed, sequence, binsize=100):
    """Agreement of generated with observed molecules in mean accessibility and co-accessibility.

    Both are (n, 2, length) arrays on the same scale (observed likelihoods divided by 256); only
    channel 0 and the A/T bases of the sequence are used. The estimates are more stable with many
    molecules, for example 500 generated molecules and up to 300 observed reads per window.

    - r_mean: Pearson r between the mean accessibility of the two sets at each A/T base;
    - r_coacc: the co-accessibility maps of the two sets (`binsize` bins, A/T bases only) are
      compared row by row, as the mean Pearson r between corresponding rows (rows with more than 30%
      valid entries; each row includes its diagonal);
    - ceiling_mean, ceiling_coacc: the largest r the finite sets allow. Each set is split into two
      consecutive halves, the halves are correlated, the correlation is Spearman-Brown corrected to
      a reliability rho = 2r / (1 + r), and the ceiling is sqrt(rho_generated * rho_observed). The
      efficiency is r / ceiling.
    """
    at = at_mask(sequence)
    summaries = {"mean": (lambda x: x[:, at].mean(axis=0), lambda a, b: np.corrcoef(a, b)[0, 1]),
                 "coacc": (lambda x: coaccessibility(x, binsize, sequence), _row_correlation)}
    out = {}
    for name, (summary, corr) in summaries.items():
        g, o = np.asarray(generated, dtype=float)[:, 0], np.asarray(observed, dtype=float)[:, 0]
        reliability = []
        for x in (g, o):
            h = len(x) // 2
            r = corr(summary(x[:h]), summary(x[h:2 * h]))
            reliability.append(2 * r / (1 + r))
        out[f"r_{name}"] = corr(summary(g), summary(o))
        out[f"ceiling_{name}"] = float(np.sqrt(reliability[0] * reliability[1]))
        out[f"efficiency_{name}"] = out[f"r_{name}"] / out[f"ceiling_{name}"]
    return out


def molecule_means(x, sequence):
    """Per molecule: mean m6A over A/T bases, mean methylation over CpG bases, and the fraction of
    CpG bases called methylated (signal >= 0.5)."""
    cpg = x[:, 1, cpg_mask(sequence)]
    return pd.DataFrame({"m6a": x[:, 0, at_mask(sequence)].mean(axis=1), "cpg": cpg.mean(axis=1),
                         "cpg_methylated": (cpg >= 0.5).mean(axis=1)})


# ----------------------------------------------------------------- panels

def clean(ax):
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0, labelsize=9.5, labelbottom=False)
    for label in ax.get_yticklabels():
        label.set_fontfamily("monospace")


def gutter_label(fig, ax, text):
    """Rotated panel name in the left gutter."""
    box = ax.get_position()
    fig.text(0.17 / fig.get_figwidth(), (box.y0 + box.y1) / 2, text, rotation=90, ha="center", va="center",
             fontsize=11, color=MUTED)


def note(ax, text):
    """Muted note at the top right of a panel."""
    ax.text(1, 1.02, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, color=MUTED)


def draw_ruler(ax, start, end):
    """A strip of round genomic coordinates above the panels."""
    span = end - start
    step = next(s for s in (100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 20000, 50000) if span / s <= 6)
    ticks = np.arange(np.ceil(start / step) * step, end, step)
    ticks = ticks[(ticks - start > 0.04 * span) & (end - ticks > 0.04 * span)]
    ax.set_ylim(0, 1)
    ax.vlines(ticks, 0, 0.3, color=BORDER, lw=1)
    ax.axhline(0, color=BORDER, lw=1)
    for t in ticks:
        ax.text(t, 0.45, f"{int(t):,}", ha="center", va="bottom", fontsize=9.5, color=MUTED, family="monospace")
    ax.set_yticks([])


def gradient_fill(ax, x, y, color, top_alpha=0.35):
    """Fill under a curve with a vertical fade."""
    rgb = to_rgb(color)
    image = np.zeros((64, 1, 4))
    image[:, 0, :3] = rgb
    image[:, 0, 3] = np.linspace(top_alpha, 0.02, 64)
    y = np.nan_to_num(y)
    im = ax.imshow(image, aspect="auto", extent=(x[0], x[-1], 0, max(y.max(), 1e-6)), origin="upper", zorder=1)
    clip = ax.fill_between(x, y, 0, color="none", lw=0)
    im.set_clip_path(clip.get_paths()[0], transform=ax.transData)


def draw_mean(ax, tracks, start, binsize, top=1.0):
    """tracks: list of (values, color, label); the first gets the gradient fill, all get a line."""
    pos = start + (np.arange(len(tracks[0][0])) + 0.5) * binsize
    for y in (0, top / 2, top):
        ax.axhline(y, color=RULE, lw=1, zorder=0)
    gradient_fill(ax, pos, tracks[0][0], MODEL)
    for values, color, label in tracks:
        ax.plot(pos, values, color=color, lw=1.5, label=label, zorder=3, solid_joinstyle="round")
    ax.set_ylim(0, top)
    ax.set_yticks([0, top / 2, top], ["0.0", f"{top / 2:.1f}", f"{top:.1f}"])
    if len(tracks) > 1:
        ax.legend(frameon=False, fontsize=9.5, loc="upper right", ncol=len(tracks), handlelength=1.4)


def draw_molecules(ax, m6a, start, binsize=5, gap=0.2):
    """One row per molecule: a gray row with m6A in purple (max over `binsize` bp per column) and a
    hairline gap between rows."""
    n, length = m6a.shape
    a = np.clip(m6a[:, :length // binsize * binsize].reshape(n, -1, binsize).max(axis=2), 0, 1)[..., None]
    rows = (1 - a) * np.array(to_rgb(SPAN)) + a * np.array(to_rgb(M6A))
    k = 5
    rgb = np.ones((n * k, rows.shape[1], 3))                     # white gap rows
    data_rows = np.repeat(rows, k - 1, axis=0)
    idx = np.arange(n * k).reshape(n, k)[:, :k - 1].ravel()
    rgb[idx] = data_rows
    ax.imshow(rgb, aspect="auto", interpolation="nearest", extent=(start, start + length, n, 0))
    ax.set_yticks([])


def draw_triangle(ax, m, start, binsize, max_separation, cmap=fall, vmin=0, vmax=0.8):
    """Co-accessibility as a triangle below the molecules: bin pairs at their midpoint (x) and
    their separation (y, downwards)."""
    n = len(m)
    edges = start + binsize * np.arange(n + 1)
    a, b = np.meshgrid(edges, edges, indexing="ij")
    upper = np.ma.masked_where(np.tril(np.ones((n, n), dtype=bool), -1) | np.isnan(m), m)
    image = ax.pcolormesh((a + b) / 2, b - a, upper, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat", rasterized=True)
    ax.set_ylim(max_separation, 0)
    step = 2000 if max_separation > 4000 else 1000
    ticks = np.arange(0, max_separation + 1, step)
    ax.set_yticks(ticks, ["0"] + [f"{t // 1000} kb" for t in ticks[1:]])
    return image


def color_key(fig, ax, image, low, high):
    """Small horizontal key at the top right of a panel."""
    box = ax.get_position()
    cax = fig.add_axes([box.x1 - 0.1, box.y1 + 0.012, 0.085, 0.012])
    fig.colorbar(image, cax=cax, orientation="horizontal")
    cax.set_xticks([])
    for side in cax.spines.values():
        side.set_visible(False)
    cax.text(-0.08, 0.5, low, transform=cax.transAxes, ha="right", va="center", fontsize=9, color=MUTED, family="monospace")
    cax.text(1.08, 0.5, high, transform=cax.transAxes, ha="left", va="center", fontsize=9, color=MUTED, family="monospace")


class Layout:
    """Stack panels of given heights (inches) under each other in a 10 inch wide figure, with a
    gutter for rotated labels on the left."""

    def __init__(self, heights, gap=0.34, top=0.55, bottom=0.18, left=1.0, right=0.3, width=10):
        self.width, self.left, self.right = width, left, right
        self.height = top + sum(heights) + gap * (len(heights) - 1) + bottom
        self.fig = plt.figure(figsize=(width, self.height))
        self.axes, y = [], self.height - top
        for h in heights:
            y -= h
            self.axes.append(self.fig.add_axes([left / width, y / self.height, (width - left - right) / width, h / self.height]))
            y -= gap

    def title(self, text):
        self.fig.text(self.left / self.width, 1 - 0.2 / self.height, text, family="monospace", fontsize=11, color=INK, va="top")

    def save(self, path):
        for p in [path] if isinstance(path, str) else path:
            self.fig.savefig(p, dpi=DPI)
        plt.close(self.fig)


# ------------------------------------------------------------------ figures

def plot_region(x, start, sequence=None, chrom="", path=None, binsize=20, max_molecules=200, max_separation=6000):
    """Mean accessibility, single molecules, and co-accessibility for one region. `path` is a file
    name or a list of them; the format follows the extension (.png, .pdf, or .svg)."""
    n, _, length = x.shape
    shown = x[:max_molecules, 0]
    max_separation = min(max_separation, length)
    axis_width = 10 - 1.0 - 0.3
    with plt.rc_context(STYLE):
        lay = Layout([0.26, 1.1, float(np.clip(len(shown) * 0.022, 1.3, 3.2)), axis_width * max_separation / length / 2])
        ax_ruler, ax_mean, ax_mol, ax_tri = lay.axes
        if chrom:
            lay.title(f"{chrom}:{start:,}–{start + length:,}")
        draw_mean(ax_mean, [(accessibility_track(x, sequence, binsize), INK, "mean m6A")], start, binsize)
        draw_molecules(ax_mol, shown, start)
        image = draw_triangle(ax_tri, coaccessibility(x[:, 0], binsize, sequence), start, binsize, max_separation)
        for ax in lay.axes:
            ax.set_xlim(start, start + length)
            clean(ax)
        draw_ruler(ax_ruler, start, start + length)
        gutter_label(lay.fig, ax_mean, "Mean accessibility")
        gutter_label(lay.fig, ax_mol, "Single molecules")
        gutter_label(lay.fig, ax_tri, "Co-accessibility")
        note(ax_mean, "m6A per A/T · 20 bp bins" if sequence is not None else "m6A · 20 bp bins")
        note(ax_mol, f"n = {n}" + (f", first {len(shown)} shown" if len(shown) < n else ""))
        ax_tri.text(0, 1.02, f"Pearson r of binned m6A{' per A/T' if sequence is not None else ''} · {binsize} bp bins", transform=ax_tri.transAxes, fontsize=9.5, color=MUTED, va="bottom")
        color_key(lay.fig, ax_tri, image, "0", "0.8")
        if path:
            lay.save(path)
    return lay.fig


def plot_variant(x_ref, x_alt, start, seq_ref, seq_alt, variant, label="", chrom="", path=None, binsize=20,
                 max_molecules=100, max_separation=3000):
    """REF and ALT molecules for a variant at `variant` (0-based genomic position): overlaid means,
    their difference, both sets of molecules, and REF, ALT, and ALT − REF co-accessibility. `path` is
    a file name or a list of them, as for `plot_region`."""
    length = x_ref.shape[2]
    axis_width = 10 - 1.0 - 0.3
    tri = axis_width * max_separation / length / 2
    mol = float(np.clip(min(len(x_ref), max_molecules) * 0.018, 1.0, 1.8))
    ref_track = accessibility_track(x_ref, seq_ref, binsize)
    alt_track = accessibility_track(x_alt, seq_alt, binsize)
    c_ref = coaccessibility(x_ref[:, 0], binsize, seq_ref)
    c_alt = coaccessibility(x_alt[:, 0], binsize, seq_alt)
    with plt.rc_context(STYLE):
        lay = Layout([0.26, 1.1, 0.6, mol, mol, tri, tri, tri])
        ax_ruler, ax_mean, ax_diff, ax_ref, ax_alt, t_ref, t_alt, t_diff = lay.axes
        lay.title(f"{chrom}:{variant + 1:,} {label}".strip())
        draw_mean(ax_mean, [(alt_track, MODEL, "ALT"), (ref_track, REF, "REF")], start, binsize)
        diff = alt_track - ref_track
        pos = start + (np.arange(len(diff)) + 0.5) * binsize
        ax_diff.axhline(0, color=BORDER, lw=1)
        ax_diff.bar(pos, diff, width=binsize, color=np.where(diff > 0, "#e34948", "#2a78d6"), lw=0)
        lim = max(0.05, np.ceil(np.nanmax(np.abs(diff)) * 20) / 20)
        ax_diff.set_ylim(-lim, lim)
        ax_diff.set_yticks([-lim, 0, lim], [f"{-lim:.2f}", "0", f"{lim:.2f}"])
        draw_molecules(ax_ref, x_ref[:max_molecules, 0], start)
        draw_molecules(ax_alt, x_alt[:max_molecules, 0], start)
        image = draw_triangle(t_ref, c_ref, start, binsize, max_separation)
        draw_triangle(t_alt, c_alt, start, binsize, max_separation)
        image_diff = draw_triangle(t_diff, c_alt - c_ref, start, binsize, max_separation, cmap=diverging, vmin=-0.2, vmax=0.2)
        for ax in lay.axes:
            ax.set_xlim(start, start + length)
            clean(ax)
            if ax is not ax_ruler:
                ax.axvline(variant + 0.5, color=DANGER, lw=0.9, alpha=0.8)
        draw_ruler(ax_ruler, start, start + length)
        for ax, text in [(ax_mean, "Mean accessibility"), (ax_diff, "ALT − REF"), (ax_ref, "Molecules\nREF"), (ax_alt, "Molecules\nALT"),
                         (t_ref, "Co-accessibility\nREF"), (t_alt, "Co-accessibility\nALT"), (t_diff, "Co-accessibility\nALT − REF")]:
            gutter_label(lay.fig, ax, text)
        note(ax_mean, "m6A per A/T · 20 bp bins")
        note(ax_ref, f"n = {len(x_ref)}")
        note(ax_alt, f"n = {len(x_alt)}")
        t_ref.text(0, 1.02, f"Pearson r of binned m6A per A/T · {binsize} bp bins", transform=t_ref.transAxes, fontsize=9.5, color=MUTED, va="bottom")
        color_key(lay.fig, t_ref, image, "0", "0.8")
        color_key(lay.fig, t_diff, image_diff, "-0.2", "0.2")
        if path:
            lay.save(path)
    return lay.fig


def write_summary(x, sequence, start, prefix, binsize=20, chrom="", formats=("png",)):
    """Write <prefix>.mean.csv, .molecules.csv, .coaccessibility.npy, and the figure in each of
    `formats` (png, pdf, or svg)."""
    mean = pd.DataFrame({"position": start + np.arange(x.shape[2]), "m6a": x[:, 0].mean(axis=0),
                         "cpg": x[:, 1].mean(axis=0)})
    if sequence is not None:
        mean.insert(1, "base", list(to_bases(sequence)))
        molecule_means(x, sequence).to_csv(f"{prefix}.molecules.csv", index_label="molecule")
    mean.to_csv(f"{prefix}.mean.csv", index=False)
    np.save(f"{prefix}.coaccessibility.npy", coaccessibility(x[:, 0], binsize, sequence))
    plot_region(x, start, sequence, chrom, [f"{prefix}.{f}" for f in formats], binsize)
