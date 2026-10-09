"""Molecules as aligned reads, for Fiber-seq tools such as FiberHMM.

Each molecule becomes one read over the input sequence, aligned to chrom:start without gaps. m6A
is stored in MM/ML tags at every A (A+a) and T (T-a), and CpG methylation at the C of every CpG
(C+m), as the mean of the values on its two bases. Values are clipped to [0, 1] and scaled to ML
probabilities of 0 to 255.
"""
from array import array

import numpy as np

BASES = np.array(list("ACGT"))


def read_bases(sequence):
    """Bases of a one-hot sequence, with N where no base is set."""
    sequence = np.asarray(sequence)
    bases = BASES[sequence.argmax(axis=1)]
    bases[sequence.sum(axis=1) < 0.5] = "N"
    return bases


def mod_tags(bases, x):
    """MM tag and ML values for one molecule x of shape (2, length)."""
    groups, ml = [], []
    cpg = np.flatnonzero((bases[:-1] == "C") & (bases[1:] == "G"))
    calls = [("A+a", bases == "A", np.flatnonzero(bases == "A"), x[0]),
             ("T-a", bases == "T", np.flatnonzero(bases == "T"), x[0]),
             ("C+m", bases == "C", cpg, None)]
    for code, is_base, pos, values in calls:
        if len(pos) == 0:
            continue
        # MM counts how many bases of this type are skipped before each listed one
        rank = np.cumsum(is_base)[pos] - 1
        skips = np.diff(rank, prepend=-1) - 1
        groups.append(f"{code}.," + ",".join(map(str, skips)))
        p = values[pos] if values is not None else (x[1, pos] + x[1, pos + 1]) / 2
        ml.append(np.rint(np.clip(p, 0, 1) * 255).astype(np.uint8))
    return ";".join(groups) + ";", np.concatenate(ml) if ml else np.zeros(0, np.uint8)


def write_bam(x, sequence, chrom, start, path, names=None, chrom_length=None):
    """Write molecules x of shape (n, 2, length) as reads aligned at chrom:start (0-based), with
    m6A and CpG methylation in MM/ML tags, and index the BAM. `chrom_length` goes into the header
    (by default, the end of the region)."""
    try:
        import pysam
    except ImportError as e:
        raise ImportError("writing a BAM needs pysam: pip install pysam") from e
    x = np.asarray(x, dtype=np.float32)
    n, _, length = x.shape
    bases = read_bases(sequence)
    seq = "".join(bases)
    names = names or [f"mola_{i}" for i in range(n)]
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": chrom, "LN": int(chrom_length or start + length)}],
              "PG": [{"ID": "mola", "PN": "mola"}]}
    quality = pysam.qualitystring_to_array("I" * length)
    with pysam.AlignmentFile(path, "wb", header=header) as out:
        for name, molecule in zip(names, x):
            mm, ml = mod_tags(bases, molecule)
            read = pysam.AlignedSegment(out.header)
            read.query_name = name
            read.query_sequence = seq
            read.flag = 0
            read.reference_id = 0
            read.reference_start = int(start)
            read.mapping_quality = 60
            read.cigartuples = [(0, length)]
            read.query_qualities = quality
            read.set_tag("MM", mm, value_type="Z")
            read.set_tag("ML", array("B", ml.tolist()))
            read.set_tag("MN", length, value_type="i")
            out.write(read)
    pysam.index(str(path))
