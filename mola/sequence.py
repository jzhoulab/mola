"""Reading and editing input sequences. Sequences are one-hot arrays of shape (length, 4), in A, C, G, T order."""
import numpy as np
import pyfaidx


BASES = "ACGT"

# one-hot rows for every byte: A, C, G, T (either case) and 0.25 everywhere for N or any other letter
ENCODING = np.full((256, 4), 0.25, dtype=np.float32)
for i, b in enumerate(BASES):
    ENCODING[ord(b)] = ENCODING[ord(b.lower())] = np.eye(4, dtype=np.float32)[i]


class Genome:
    """A FASTA file with one-hot access, matching selene-sdk's Genome, which Mola was trained with.

    Coordinates outside a chromosome give an empty sequence. With a blacklist (a BED file),
    any window that overlaps it also comes back empty, so training skips it.
    """

    def __init__(self, path, blacklist=None):
        self.fasta = pyfaidx.Fasta(path, as_raw=True)
        self.blacklist = {}
        if blacklist:
            for line in open(blacklist):
                chrom, start, end = line.split()[:3]
                self.blacklist.setdefault(chrom, []).append((int(start), int(end)))

    def get_chr_lens(self):
        return [(chrom, len(self.fasta[chrom])) for chrom in sorted(self.fasta.keys())]

    def get_sequence_from_coords(self, chrom, start, end):
        if chrom not in self.fasta or start < 0 or start >= end or end > len(self.fasta[chrom]):
            return ""
        if any(s < end and e > start for s, e in self.blacklist.get(chrom, [])):
            return ""
        return self.fasta[chrom][start:end]

    def get_encoding_from_coords(self, chrom, start, end):
        sequence = self.get_sequence_from_coords(chrom, start, end)
        return ENCODING[np.frombuffer(sequence.encode(), dtype=np.uint8)]


def load_genome(path="resources/hg38.fa", blacklist=None):
    return Genome(path, blacklist)


def parse_region(region):
    """'chr19:47509000-47519000' -> ('chr19', 47509000, 47519000). 0-based, end-exclusive."""
    chrom, coords = region.split(":")
    start, end = coords.replace(",", "").split("-")
    return chrom, int(start), int(end)


def one_hot(bases):
    return np.eye(4, dtype=np.float32)[[BASES.index(b) for b in bases.upper()]]


def to_bases(sequence):
    return "".join(np.array(list(BASES))[np.asarray(sequence).argmax(axis=1)])


def insert(sequence, motif, position):
    """Copy of the sequence with `motif` written over the bases starting at `position`.

    This is a replacement, not an insertion, so the length stays the same; in silico motif
    insertions replace the reference bases with the motif consensus in this way.
    """
    edited = np.array(sequence, dtype=np.float32)
    edited[position:position + len(motif)] = one_hot(motif)
    return edited


def cpg_mask(sequence):
    """True at the C and the G of every CpG."""
    bases = np.asarray(sequence).argmax(axis=1)
    cg = (bases[:-1] == 1) & (bases[1:] == 2)
    mask = np.zeros(len(bases), dtype=bool)
    mask[:-1] |= cg
    mask[1:] |= cg
    return mask


def at_mask(sequence):
    """True at A and T bases, the only positions where m6A can be measured."""
    return np.asarray(sequence)[:, [0, 3]].sum(axis=1) > 0.5


def random_windows(genome, n, chrom="chr10", length=10000, seed=0):
    """n random windows without N bases, as a list of (start, sequence). chr10 was held out from training."""
    rng = np.random.default_rng(seed)
    size = dict(genome.get_chr_lens())[chrom]
    windows = []
    while len(windows) < n:
        start = int(rng.integers(0, size - length))
        seq = genome.get_encoding_from_coords(chrom, start, start + length)
        if (seq.max(axis=1) == 1).all():
            windows.append((start, seq))
    return windows
