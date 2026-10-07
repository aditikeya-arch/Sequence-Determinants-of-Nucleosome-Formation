

"""
Code 7 Figure 2-12 GC and poly dAdT analysis.py
================================================

WHAT THIS SCRIPT DOES
---------------------
Generates the data underlying Figure 2.12.

Figure 2.12a:
    Mean GC content of the central 113-bp variable region
    for SELEX rounds R1-R6.

    Two quantities are calculated:
        1. unweighted mean across unique sequences
        2. CPM-weighted mean across sequences

Figure 2.12b:
    Fraction of bases belonging to poly(dA:dT) tracts,
    where a tract is defined as an uninterrupted run of
    >=5 A bases OR >=5 T bases.

    The observed SELEX values are compared with random
    sequences matched only for the unweighted GC content
    of the corresponding SELEX round.

    For each round:
        10 independent simulations
        20,000 sequences per simulation
        113 bp per sequence


INPUT FILES
-----------
R1.csv
R2.csv
R3.csv
R4.csv
R5.csv
R6.csv

Each file must contain:
    Sequence
    Counts Per Million


OTHER SCRIPTS THAT MUST BE RUN FIRST
------------------------------------
None.


IMPORTANT
---------
The original SELEX sequences are 147 bp:

    17-bp constant flank
    113-bp variable region
    17-bp constant flank

Only the central 113 bp are analysed here.


OUTPUT VARIABLES
----------------
figure_2_12a
figure_2_12b

gc_unweighted
gc_weighted

poly_dat_observed

poly_dat_random_mean
poly_dat_random_sd

poly_dat_random_replicates
"""


import numpy as np
import pandas as pd


# ============================================================
# 1. General settings
# ============================================================

round_names = [
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6"
]


# Random-number seed added for reproducibility.
#
# The thesis does not specify the seed used in the original
# analysis.

rng = np.random.default_rng(1)


# ============================================================
# 2. Extract central 113-bp variable region
# ============================================================

def central_113(sequence):

    sequence = str(sequence)

    if len(sequence) != 147:

        raise ValueError(
            "Expected a 147-bp SELEX sequence, "
            f"but found length {len(sequence)}."
        )

    return sequence[17:130]


# ============================================================
# 3. GC fraction of one sequence
# ============================================================

def gc_fraction(sequence):

    sequence = sequence.upper()

    return (
        sequence.count("G")
        + sequence.count("C")
    ) / len(sequence)


# ============================================================
# 4. Number of bases belonging to poly(dA:dT) tracts
# ============================================================

def count_poly_dat_bases(sequence, minimum_run=5):

    """
    Count bases belonging to uninterrupted runs of >=5 A
    or >=5 T.

    Examples:

        AAAAA      -> 5 bases
        AAAAAA     -> 6 bases
        TTTTT      -> 5 bases
        AAAATTTT   -> 0 bases

    A-run and T-run tracts are treated separately.
    """

    sequence = sequence.upper()

    total = 0

    i = 0

    while i < len(sequence):

        base = sequence[i]

        if base in ("A", "T"):

            j = i + 1

            while (
                j < len(sequence)
                and sequence[j] == base
            ):
                j += 1

            run_length = j - i

            if run_length >= minimum_run:

                total += run_length

            i = j

        else:

            i += 1

    return total


# ============================================================
# 5. Fraction of bases in poly(dA:dT) tracts for a sequence set
# ============================================================

def poly_dat_fraction(sequences, weights=None):

    """
    Fraction of all bases in the sequence pool that belong
    to poly(dA:dT) tracts.

    If weights are supplied, each sequence contributes in
    proportion to its weight.
    """

    sequences = list(sequences)

    poly_counts = np.array(
        [
            count_poly_dat_bases(s)
            for s in sequences
        ],
        dtype=float
    )

    lengths = np.array(
        [
            len(s)
            for s in sequences
        ],
        dtype=float
    )

    if weights is None:

        return (
            np.sum(poly_counts)
            / np.sum(lengths)
        )

    weights = np.asarray(
        weights,
        dtype=float
    )

    return (
        np.sum(poly_counts * weights)
        / np.sum(lengths * weights)
    )


# ============================================================
# 6. Generate GC-matched random DNA
# ============================================================

def generate_gc_matched_sequences(
    gc_fraction_target,
    n_sequences,
    sequence_length,
    rng
):

    """
    Generate random DNA under the sole constraint of the
    specified overall GC fraction.

    Bases are sampled independently with:

        P(G) = GC / 2
        P(C) = GC / 2
        P(A) = (1-GC) / 2
        P(T) = (1-GC) / 2

    Thus there is no sequence structure beyond the specified
    base composition.
    """

    probabilities = [

        (1 - gc_fraction_target) / 2,   # A
        gc_fraction_target / 2,         # C
        gc_fraction_target / 2,         # G
        (1 - gc_fraction_target) / 2    # T
    ]

    bases = np.array(
        ["A", "C", "G", "T"]
    )

    random_array = rng.choice(

        bases,

        size=(
            n_sequences,
            sequence_length
        ),

        p=probabilities
    )

    return [
        "".join(row)
        for row in random_array
    ]


# ============================================================
# 7. Containers for results
# ============================================================

gc_unweighted = {}
gc_weighted = {}

poly_dat_observed = {}

poly_dat_random_replicates = {}

poly_dat_random_mean = {}
poly_dat_random_sd = {}


# ============================================================
# 8. Analyse each SELEX round
# ============================================================

for round_name in round_names:

    print(
        "Processing",
        round_name
    )

    # --------------------------------------------------------
    # Read data
    # --------------------------------------------------------

    df = pd.read_csv(
        f"{round_name}.csv"
    )


    # --------------------------------------------------------
    # Extract central 113 bp
    # --------------------------------------------------------

    sequences_113 = [

        central_113(s)

        for s in df[
            "Sequence"
        ]
    ]


    cpm = df[
        "Counts Per Million"
    ].to_numpy(
        dtype=float
    )


    # --------------------------------------------------------
    # Figure 2.12a
    #
    # GC content
    # --------------------------------------------------------

    sequence_gc = np.array(
        [
            gc_fraction(s)
            for s in sequences_113
        ]
    )


    # Unweighted mean across UNIQUE sequences.

    gc_unweighted[
        round_name
    ] = np.mean(
        sequence_gc
    )


    # CPM-weighted mean.

    gc_weighted[
        round_name
    ] = np.average(

        sequence_gc,

        weights=cpm
    )


    # --------------------------------------------------------
    # Figure 2.12b
    #
    # Observed poly(dA:dT)
    # --------------------------------------------------------

    #
    # The thesis describes the quantity as the fraction of
    # bases "among all reads" belonging to poly(dA:dT) tracts.
    #
    # Since the CSV contains unique sequences and CPM, CPM
    # weighting reconstructs the contribution from all reads.
    #

    poly_dat_observed[
        round_name
    ] = poly_dat_fraction(

        sequences_113,

        weights=cpm
    )


    # --------------------------------------------------------
    # GC-matched random controls
    # --------------------------------------------------------

    #
    # IMPORTANT:
    #
    # The thesis explicitly states that the random sequences
    # were matched to the UNWEIGHTED GC content shown in
    # Figure 2.12a.
    #

    target_gc = gc_unweighted[
        round_name
    ]


    replicate_values = []


    for replicate in range(10):

        random_sequences = (
            generate_gc_matched_sequences(

                gc_fraction_target=target_gc,

                n_sequences=20000,

                sequence_length=113,

                rng=rng
            )
        )


        random_poly_dat = (
            poly_dat_fraction(
                random_sequences
            )
        )


        replicate_values.append(
            random_poly_dat
        )


    replicate_values = np.array(
        replicate_values
    )


    poly_dat_random_replicates[
        round_name
    ] = replicate_values


    poly_dat_random_mean[
        round_name
    ] = np.mean(
        replicate_values
    )


    poly_dat_random_sd[
        round_name
    ] = np.std(
        replicate_values,
        ddof=1
    )


# ============================================================
# 9. Figure 2.12a output
# ============================================================

figure_2_12a = pd.DataFrame({

    "Round":
        round_names,

    "GC_unweighted":
        [
            gc_unweighted[r]
            for r in round_names
        ],

    "GC_CPM_weighted":
        [
            gc_weighted[r]
            for r in round_names
        ]
})


print(
    "\nFigure 2.12a:"
)

print(
    figure_2_12a
)


# ============================================================
# 10. Figure 2.12b output
# ============================================================

figure_2_12b = pd.DataFrame({

    "Round":
        round_names,

    "Poly_dAdT_SELEX":
        [
            poly_dat_observed[r]
            for r in round_names
        ],

    "Poly_dAdT_random_mean":
        [
            poly_dat_random_mean[r]
            for r in round_names
        ],

    "Poly_dAdT_random_SD":
        [
            poly_dat_random_sd[r]
            for r in round_names
        ]
})


print(
    "\nFigure 2.12b:"
)

print(
    figure_2_12b
)


# ============================================================
# FIGURE OUTPUT SUMMARY
# ============================================================

# Figure 2.12a:
#
#     figure_2_12a["Round"]
#     figure_2_12a["GC_unweighted"]
#     figure_2_12a["GC_CPM_weighted"]
#
#
# Figure 2.12b:
#
#     figure_2_12b["Round"]
#     figure_2_12b["Poly_dAdT_SELEX"]
#     figure_2_12b["Poly_dAdT_random_mean"]
#     figure_2_12b["Poly_dAdT_random_SD"]
#
# Random SD gives the error bars for the GC-matched control.