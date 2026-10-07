

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

N_ROUNDS = 7
N_SAMPLE = 10_000
SEQ_LENGTH = 113
MIN_POLY_RUN = 5


# ============================================================
# 1. LOAD THE 10,000 SAMPLED SEQUENCES FOR EACH ROUND
# ============================================================

seqs = []

for r in range(N_ROUNDS):

    filename = f"seq{r}.pkl"

    with open(filename, "rb") as f:
        sequences = pickle.load(f)

    # Sanity checks
    if len(sequences) != N_SAMPLE:
        raise ValueError(
            f"{filename} contains {len(sequences)} sequences, "
            f"expected {N_SAMPLE}."
        )

    if not all(len(s) == SEQ_LENGTH for s in sequences):
        raise ValueError(
            f"Not all sequences in {filename} are {SEQ_LENGTH} bp."
        )

    seqs.append(sequences)

    print(f"Loaded R{r}: {len(sequences):,} sequences")


seq0, seq1, seq2, seq3, seq4, seq5, seq6 = seqs


# ============================================================
# 2. GC CONTENT
# ============================================================

def gc_fraction(sequence):
    """
    Fraction of bases in a sequence that are G or C.
    """

    sequence = sequence.upper()

    return (
        sequence.count("G") +
        sequence.count("C")
    ) / len(sequence)


# ============================================================
# 3. POLY(dA:dT) CONTENT
# ============================================================

def count_poly_dat_bases(sequence, minimum_run=5):
    """
    Count bases belonging to uninterrupted runs of >=5
    consecutive A's OR >=5 consecutive T's.

    Examples:

        AAAAA       -> 5
        AAAAAA      -> 6
        TTTTT       -> 5
        TTTTTTT     -> 7
        AAAATTTT    -> 0
        AAAAATTTTT  -> 10
    """

    sequence = sequence.upper()

    total = 0
    i = 0

    while i < len(sequence):

        base = sequence[i]

        if base in ("A", "T"):

            j = i + 1

            # Find end of this identical-base run
            while j < len(sequence) and sequence[j] == base:
                j += 1

            run_length = j - i

            if run_length >= minimum_run:
                total += run_length

            i = j

        else:
            i += 1

    return total


def poly_dat_fraction(sequence, minimum_run=5):
    """
    Fraction of bases in an individual sequence belonging
    to poly(dA:dT) tracts.
    """

    return (
        count_poly_dat_bases(
            sequence,
            minimum_run=minimum_run
        )
        / len(sequence)
    )


# ============================================================
# 4. CALCULATE MEAN VALUES FOR EACH ROUND
# ============================================================

mean_gc = []
mean_poly_dat = []

for r, sequences in enumerate(seqs):

    print(f"\nProcessing R{r}...")


    # --------------------------------------------------------
    # GC content of each of the 10,000 sequences
    # --------------------------------------------------------

    gc_values = np.array([
        gc_fraction(s)
        for s in sequences
    ])


    # --------------------------------------------------------
    # poly(dA:dT) content of each of the 10,000 sequences
    # --------------------------------------------------------

    poly_values = np.array([
        poly_dat_fraction(
            s,
            minimum_run=MIN_POLY_RUN
        )
        for s in sequences
    ])


    # Every sequence gets equal weight
    mean_gc.append(
        gc_values.mean()
    )

    mean_poly_dat.append(
        poly_values.mean()
    )


    print(
        f"  Mean GC content:       "
        f"{mean_gc[-1]:.4f}"
    )

    print(
        f"  Mean poly(dA:dT):      "
        f"{mean_poly_dat[-1]:.4f}"
    )


mean_gc = np.array(mean_gc)
mean_poly_dat = np.array(mean_poly_dat)


# ============================================================
# 5. PUT RESULTS INTO A DATAFRAME
# ============================================================

results = pd.DataFrame({

    "Round": [
        f"R{r}"
        for r in range(N_ROUNDS)
    ],

    "Mean_GC_fraction":
        mean_gc,

    "Mean_poly_dAdT_fraction":
        mean_poly_dat

})


print("\n========================================")
print("RESULTS")
print("========================================")

print(results.to_string(index=False))


# ============================================================
# 6. SAVE NUMERICAL RESULTS
# ============================================================

results.to_csv(
    "GC_poly_dAdT_10k_unweighted.csv",
    index=False
)


# ============================================================
# 7. PLOT GC CONTENT
# ============================================================

rounds = np.arange(N_ROUNDS)


fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(
    rounds,
    mean_gc,
    marker="o",
    linewidth=2
)

ax.set_xticks(rounds)

ax.set_xticklabels([
    f"R{r}"
    for r in rounds
])

ax.set_xlabel("SELEX round")

ax.set_ylabel(
    "Mean GC fraction"
)

ax.set_title(
    "GC content across SELEX rounds"
)

ax.grid(alpha=0.25)

fig.tight_layout()

fig.savefig(
    "GC_content_10k_unweighted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 8. PLOT POLY(dA:dT) CONTENT
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(
    rounds,
    mean_poly_dat,
    marker="o",
    linewidth=2
)

ax.set_xticks(rounds)

ax.set_xticklabels([
    f"R{r}"
    for r in rounds
])

ax.set_xlabel("SELEX round")

ax.set_ylabel(
    "Mean poly(dA:dT) fraction"
)

ax.set_title(
    "Poly(dA:dT) content across SELEX rounds"
)

ax.grid(alpha=0.25)

fig.tight_layout()

fig.savefig(
    "poly_dAdT_content_10k_unweighted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()