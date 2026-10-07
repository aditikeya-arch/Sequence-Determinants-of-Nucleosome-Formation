

"""
Code 8 Figure 2-13 Dinucleotide enrichment analysis.py
=======================================================

WHAT THIS SCRIPT DOES
---------------------
Generates the data underlying Figure 2.13.

For each SELEX round:

1. Extract the central 113-bp variable region.

2. Calculate the average per-read frequency of each of the
   16 possible dinucleotides.

3. Generate GC-matched random sequences:
       5 independent replicate datasets per round
       10,000 sequences per replicate
       113 bp per sequence

4. Calculate dinucleotide frequencies in each simulated
   dataset.

5. Express observed dinucleotide frequencies as log2
   fold-change relative to round 1.

6. For the simulated datasets, calculate log2 fold-change
   for each replicate relative to ITS OWN round-1 value.

7. Calculate the mean and standard deviation of the five
   simulated log2 fold-change values.


THESIS FIGURE
-------------
Figure 2.13

Each of the 16 dinucleotides has enrichment dynamics across
R1-R6.

Observed SELEX enrichment is compared with enrichment
expected from the change in GC content alone.


INPUT FILES
-----------
R1.csv
R2.csv
R3.csv
R4.csv
R5.csv
R6.csv

Required columns:

    Sequence
    Counts Per Million


OTHER SCRIPTS THAT MUST BE RUN FIRST
------------------------------------
None.


OUTPUT VARIABLES
----------------
dinucleotide_frequency_observed

dinucleotide_log2fc_observed

dinucleotide_frequency_random

dinucleotide_log2fc_random

dinucleotide_log2fc_random_mean
dinucleotide_log2fc_random_sd

figure_2_13
"""


import itertools

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


bases = [
    "A",
    "C",
    "G",
    "T"
]


dinucleotides = [

    "".join(x)

    for x in itertools.product(
        bases,
        repeat=2
    )
]


print(
    "Dinucleotides:",
    dinucleotides
)


# Seed added for reproducibility.
#
# The thesis does not specify the original random seed.

rng = np.random.default_rng(1)


# ============================================================
# 2. Extract central 113 bp
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
# 3. GC fraction
# ============================================================

def gc_fraction(sequence):

    sequence = sequence.upper()

    return (
        sequence.count("G")
        + sequence.count("C")
    ) / len(sequence)


# ============================================================
# 4. Count all overlapping dinucleotides in one sequence
# ============================================================

def dinucleotide_counts(sequence):

    sequence = sequence.upper()

    counts = {
        dinucleotide: 0
        for dinucleotide in dinucleotides
    }

    for i in range(
        len(sequence) - 1
    ):

        dinucleotide = sequence[
            i:i+2
        ]

        if dinucleotide in counts:

            counts[
                dinucleotide
            ] += 1

    return counts


# ============================================================
# 5. Mean per-read dinucleotide frequency
# ============================================================

def mean_dinucleotide_frequency(
    sequences,
    weights=None
):

    """
    Calculate average per-read dinucleotide frequency.

    A 113-bp sequence contains 112 overlapping
    dinucleotide positions.

    For each sequence:

        frequency =
            number of occurrences / 112

    If weights are supplied, the per-sequence frequencies
    are averaged using those weights.

    CPM weighting therefore reconstructs the average over
    all reads in the SELEX pool.
    """

    sequences = list(
        sequences
    )


    frequency_matrix = np.zeros(

        (
            len(sequences),
            len(dinucleotides)
        ),

        dtype=float
    )


    for row_index, sequence in enumerate(
        sequences
    ):

        counts = dinucleotide_counts(
            sequence
        )

        denominator = (
            len(sequence) - 1
        )

        for column_index, dinucleotide in enumerate(
            dinucleotides
        ):

            frequency_matrix[
                row_index,
                column_index
            ] = (
                counts[dinucleotide]
                / denominator
            )


    if weights is None:

        mean_frequency = np.mean(

            frequency_matrix,

            axis=0
        )

    else:

        mean_frequency = np.average(

            frequency_matrix,

            axis=0,

            weights=np.asarray(
                weights,
                dtype=float
            )
        )


    return dict(
        zip(
            dinucleotides,
            mean_frequency
        )
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

    probabilities = [

        (1 - gc_fraction_target) / 2,   # A

        gc_fraction_target / 2,         # C

        gc_fraction_target / 2,         # G

        (1 - gc_fraction_target) / 2    # T
    ]


    base_array = np.array(
        ["A", "C", "G", "T"]
    )


    random_array = rng.choice(

        base_array,

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
# 7. Read all SELEX rounds
# ============================================================

round_data = {}


for round_name in round_names:

    print(
        "Reading",
        round_name
    )


    df = pd.read_csv(
        f"{round_name}.csv"
    )


    df["Sequence_113"] = [

        central_113(s)

        for s in df[
            "Sequence"
        ]
    ]


    round_data[
        round_name
    ] = df


# ============================================================
# 8. Determine UNWEIGHTED GC fraction for each round
# ============================================================

#
# As in the GC-matched control used for Figure 2.12, the
# synthetic sequences are constrained only by the observed
# GC fraction of the corresponding round.
#
# We use the unweighted mean across unique sequences.
#

round_gc = {}


for round_name in round_names:

    df = round_data[
        round_name
    ]


    sequence_gc = np.array(
        [
            gc_fraction(s)

            for s in df[
                "Sequence_113"
            ]
        ]
    )


    round_gc[
        round_name
    ] = np.mean(
        sequence_gc
    )


print(
    "\nGC fractions used for simulations:"
)

print(
    round_gc
)


# ============================================================
# 9. Observed dinucleotide frequencies
# ============================================================

#
# The thesis specifies average per-read frequencies.
#
# Because the CSV contains unique sequences and CPM values,
# CPM weighting is used to reconstruct the average across
# reads.
#

dinucleotide_frequency_observed = {}


for round_name in round_names:

    print(
        "Observed dinucleotide frequencies:",
        round_name
    )


    df = round_data[
        round_name
    ]


    dinucleotide_frequency_observed[
        round_name
    ] = mean_dinucleotide_frequency(

        sequences=df[
            "Sequence_113"
        ],

        weights=df[
            "Counts Per Million"
        ]
    )


# ============================================================
# 10. Observed log2 fold-change relative to R1
# ============================================================

dinucleotide_log2fc_observed = {}


for round_name in round_names:

    dinucleotide_log2fc_observed[
        round_name
    ] = {}


    for dinucleotide in dinucleotides:

        current_frequency = (
            dinucleotide_frequency_observed[
                round_name
            ][
                dinucleotide
            ]
        )


        r1_frequency = (
            dinucleotide_frequency_observed[
                "R1"
            ][
                dinucleotide
            ]
        )


        dinucleotide_log2fc_observed[
            round_name
        ][
            dinucleotide
        ] = np.log2(

            current_frequency
            / r1_frequency
        )


# ============================================================
# 11. Generate five GC-matched simulated datasets per round
# ============================================================

#
# Structure:
#
# dinucleotide_frequency_random[round][replicate][dinucleotide]
#

dinucleotide_frequency_random = {}


for round_name in round_names:

    print(
        "Simulating",
        round_name
    )


    dinucleotide_frequency_random[
        round_name
    ] = []


    target_gc = round_gc[
        round_name
    ]


    for replicate in range(5):

        random_sequences = (
            generate_gc_matched_sequences(

                gc_fraction_target=target_gc,

                n_sequences=10000,

                sequence_length=113,

                rng=rng
            )
        )


        frequencies = (
            mean_dinucleotide_frequency(
                random_sequences
            )
        )


        dinucleotide_frequency_random[
            round_name
        ].append(
            frequencies
        )


# ============================================================
# 12. Simulated log2 fold-change relative to R1
# ============================================================

#
# IMPORTANT:
#
# The thesis explicitly states that simulated fold-changes
# were calculated separately for each replicate relative
# to ITS OWN round-1 value.
#
# Thus:
#
# replicate 0 R2 / replicate 0 R1
# replicate 1 R2 / replicate 1 R1
# ...
# replicate 4 R2 / replicate 4 R1
#

dinucleotide_log2fc_random = {}


for round_name in round_names:

    dinucleotide_log2fc_random[
        round_name
    ] = []


    for replicate in range(5):

        replicate_result = {}


        for dinucleotide in dinucleotides:

            current_frequency = (
                dinucleotide_frequency_random[
                    round_name
                ][
                    replicate
                ][
                    dinucleotide
                ]
            )


            r1_frequency = (
                dinucleotide_frequency_random[
                    "R1"
                ][
                    replicate
                ][
                    dinucleotide
                ]
            )


            replicate_result[
                dinucleotide
            ] = np.log2(

                current_frequency
                / r1_frequency
            )


        dinucleotide_log2fc_random[
            round_name
        ].append(
            replicate_result
        )


# ============================================================
# 13. Mean and SD of simulated log2 fold-change
# ============================================================

dinucleotide_log2fc_random_mean = {}

dinucleotide_log2fc_random_sd = {}


for round_name in round_names:

    dinucleotide_log2fc_random_mean[
        round_name
    ] = {}


    dinucleotide_log2fc_random_sd[
        round_name
    ] = {}


    for dinucleotide in dinucleotides:

        values = np.array([

            dinucleotide_log2fc_random[
                round_name
            ][
                replicate
            ][
                dinucleotide
            ]

            for replicate in range(5)
        ])


        dinucleotide_log2fc_random_mean[
            round_name
        ][
            dinucleotide
        ] = np.mean(
            values
        )


        dinucleotide_log2fc_random_sd[
            round_name
        ][
            dinucleotide
        ] = np.std(
            values,
            ddof=1
        )


# ============================================================
# 14. Assemble Figure 2.13 data
# ============================================================

rows = []


for dinucleotide in dinucleotides:

    for round_name in round_names:

        rows.append({

            "Dinucleotide":
                dinucleotide,

            "Round":
                round_name,

            "Observed_log2FC":
                dinucleotide_log2fc_observed[
                    round_name
                ][
                    dinucleotide
                ],

            "GC_matched_log2FC_mean":
                dinucleotide_log2fc_random_mean[
                    round_name
                ][
                    dinucleotide
                ],

            "GC_matched_log2FC_SD":
                dinucleotide_log2fc_random_sd[
                    round_name
                ][
                    dinucleotide
                ]
        })


figure_2_13 = pd.DataFrame(
    rows
)


print(
    "\nFigure 2.13 data:"
)

print(
    figure_2_13
)


# ============================================================
# 15. Optional wide tables
# ============================================================

#
# These are convenient if the data are later passed directly
# to plotting software.
#

figure_2_13_observed = (
    figure_2_13.pivot(

        index="Round",

        columns="Dinucleotide",

        values="Observed_log2FC"
    )
)


figure_2_13_random_mean = (
    figure_2_13.pivot(

        index="Round",

        columns="Dinucleotide",

        values="GC_matched_log2FC_mean"
    )
)


figure_2_13_random_sd = (
    figure_2_13.pivot(

        index="Round",

        columns="Dinucleotide",

        values="GC_matched_log2FC_SD"
    )
)


# ============================================================
# FIGURE OUTPUT SUMMARY
# ============================================================

# Main tidy-format output:
#
#     figure_2_13
#
# columns:
#
#     Dinucleotide
#     Round
#     Observed_log2FC
#     GC_matched_log2FC_mean
#     GC_matched_log2FC_SD
#
#
# Each dinucleotide can therefore be plotted as:
#
# x:
#     Round
#
# observed:
#     Observed_log2FC
#
# GC-matched expectation:
#     GC_matched_log2FC_mean
#
# error:
#     GC_matched_log2FC_SD
#
#
# R1 should be exactly:
#
#     Observed_log2FC = 0
#
# and each individual random replicate should also have
# R1 log2FC = 0 because it is divided by itself.