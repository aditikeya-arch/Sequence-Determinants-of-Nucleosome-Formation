

"""
Code 9 Figure 2-14 Dinucleotide mechanical effect and enrichment.py
===================================================================

WHAT THIS SCRIPT DOES
---------------------
Generates the data underlying Figure 2.14.

Figure 2.14a:
    Calculates the "mechanical effect" of each of the 16
    dinucleotides on intrinsic DNA cyclisability.

    For each dinucleotide:
        - generate 500,000 random 50-bp sequences
        - sequence structure is:

              24 random bp
              + fixed dinucleotide
              + 24 random bp

        - predict intrinsic cyclisability
        - calculate the mean prediction

    A separate set of 500,000 completely random 50-bp
    sequences is used to define the baseline.

    Mechanical effect =

        mean cyclisability with fixed dinucleotide
        -
        mean cyclisability of random DNA


Figure 2.14b:
    Observed log2 enrichment of each dinucleotide in
    SELEX round 6 relative to round 1.

    These values come directly from Figure 2.13.


Figure 2.14c:
    Dinucleotide enrichment relative to the GC-matched
    random expectation from Figure 2.13.


VARIABLES THAT MUST ALREADY EXIST
---------------------------------
load_model
pred

These functions come from the external intrinsic
cyclisability prediction code used throughout the thesis.

They are the same functions used in:

    Code 1 Intrinsic cyclisability as a function of position.py
    Code 3 intrinsic cycisability of all sequences and mean profiles of all rounds.py
    Code 4 PCA analysis Fig 2-10.py
    Code 6 Figure 2-11 Native nucleosome PCA analysis.py


OTHER SCRIPTS THAT MUST BE RUN FIRST
------------------------------------
Code 8 Figure 2-13 Dinucleotide enrichment analysis.py

This script uses:

    dinucleotide_log2fc_observed
    dinucleotide_log2fc_random_mean


OUTPUT VARIABLES
----------------
dinucleotide_mechanical_effect

mechanical_effect_table

dinucleotide_order

figure_2_14

figure_2_14a
figure_2_14b
figure_2_14c


NOTES
-----
The thesis specifies 500,000 sequences per dinucleotide
and 500,000 fully random sequences for the baseline.

A random-number seed is added here for reproducibility.
The thesis does not specify the seed used in the original
analysis.

No plotting is performed in this script.
"""


import itertools

import numpy as np
import pandas as pd


# ============================================================
# 1. Settings
# ============================================================

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


N_RANDOM = 500000


# Added for reproducibility.
#
# The original seed is not specified in the thesis.

rng = np.random.default_rng(1)


# ============================================================
# 2. Check required variables from Code 8
# ============================================================

required_variables = [

    "dinucleotide_log2fc_observed",

    "dinucleotide_log2fc_random_mean"
]


for variable_name in required_variables:

    if variable_name not in globals():

        raise RuntimeError(

            f"{variable_name} is not defined. "
            "Run "
            "'Code 8 Figure 2-13 Dinucleotide enrichment analysis.py' "
            "before running this script."
        )


# ============================================================
# 3. Load intrinsic cyclisability model
# ============================================================

#
# load_model() and pred() are assumed to already be defined.
#
# These are the same external functions used in the earlier
# intrinsic cyclisability scripts.
#

model = load_model(0)


# ============================================================
# 4. Generate fully random DNA sequences
# ============================================================

def generate_random_sequences(
    n_sequences,
    sequence_length,
    rng
):

    """
    Generate DNA with equal probability of A, C, G and T
    independently at every position.
    """

    base_array = np.array(
        ["A", "C", "G", "T"]
    )


    random_array = rng.choice(

        base_array,

        size=(
            n_sequences,
            sequence_length
        )
    )


    return [

        "".join(row)

        for row in random_array
    ]


# ============================================================
# 5. Generate sequences containing a fixed dinucleotide
# ============================================================

def generate_sequences_with_fixed_dinucleotide(
    dinucleotide,
    n_sequences,
    rng
):

    """
    Generate sequences of the form:

        random 24 bp
        + fixed dinucleotide
        + random 24 bp

    Total length = 50 bp.
    """

    base_array = np.array(
        ["A", "C", "G", "T"]
    )


    left_flanks = rng.choice(

        base_array,

        size=(
            n_sequences,
            24
        )
    )


    right_flanks = rng.choice(

        base_array,

        size=(
            n_sequences,
            24
        )
    )


    sequences = [

        "".join(left)
        + dinucleotide
        + "".join(right)

        for left, right in zip(
            left_flanks,
            right_flanks
        )
    ]


    return sequences


# ============================================================
# 6. Predict mean cyclisability
# ============================================================

def mean_predicted_cyclisability(
    sequences,
    model
):

    predictions = np.asarray(

        pred(
            model,
            sequences
        ),

        dtype=float
    ).reshape(-1)


    return np.mean(
        predictions
    )


# ============================================================
# 7. Baseline from completely random 50-bp DNA
# ============================================================

print(
    "Generating 500,000 completely random 50-bp sequences..."
)


baseline_sequences = (
    generate_random_sequences(

        n_sequences=N_RANDOM,

        sequence_length=50,

        rng=rng
    )
)


print(
    "Predicting baseline intrinsic cyclisability..."
)


baseline_predictions = np.asarray(

    pred(
        model,
        baseline_sequences
    ),

    dtype=float

).reshape(-1)


baseline_mean_cyclisability = np.mean(
    baseline_predictions
)


print(
    "Baseline mean cyclisability:",
    baseline_mean_cyclisability
)


# Free the sequences because the next steps are large.

del baseline_sequences


# ============================================================
# 8. Mechanical effect of each dinucleotide
# ============================================================

dinucleotide_mean_cyclisability = {}

dinucleotide_mechanical_effect = {}


for dinucleotide in dinucleotides:

    print(
        "Processing dinucleotide:",
        dinucleotide
    )


    sequences = (
        generate_sequences_with_fixed_dinucleotide(

            dinucleotide=dinucleotide,

            n_sequences=N_RANDOM,

            rng=rng
        )
    )


    predictions = np.asarray(

        pred(
            model,
            sequences
        ),

        dtype=float

    ).reshape(-1)


    mean_cyclisability = np.mean(
        predictions
    )


    mechanical_effect = (

        mean_cyclisability
        -
        baseline_mean_cyclisability
    )


    dinucleotide_mean_cyclisability[
        dinucleotide
    ] = mean_cyclisability


    dinucleotide_mechanical_effect[
        dinucleotide
    ] = mechanical_effect


    print(
        "    mean cyclisability =",
        mean_cyclisability
    )

    print(
        "    mechanical effect =",
        mechanical_effect
    )


    # Do not retain 500,000 sequences/predictions after
    # this dinucleotide has been processed.

    del sequences
    del predictions


# ============================================================
# 9. Mechanical-effect table
# ============================================================

mechanical_effect_table = pd.DataFrame({

    "Dinucleotide":
        dinucleotides,

    "Mean_cyclisability":
        [
            dinucleotide_mean_cyclisability[d]
            for d in dinucleotides
        ],

    "Mechanical_effect":
        [
            dinucleotide_mechanical_effect[d]
            for d in dinucleotides
        ]
})


# ============================================================
# 10. Sort dinucleotides by mechanical effect
# ============================================================

#
# Figure 2.14a is described as a sorted bar plot.
#
# Panels b and c use the SAME dinucleotide ordering.
#

mechanical_effect_table = (
    mechanical_effect_table
    .sort_values(
        "Mechanical_effect"
    )
    .reset_index(
        drop=True
    )
)


dinucleotide_order = (
    mechanical_effect_table[
        "Dinucleotide"
    ].tolist()
)


print(
    "\nDinucleotide order from least to most "
    "flexibility-promoting:"
)

print(
    dinucleotide_order
)


# ============================================================
# 11. Figure 2.14a
# ============================================================

figure_2_14a = (
    mechanical_effect_table.copy()
)


# ============================================================
# 12. Figure 2.14b
#
# Observed R6 / R1 enrichment
# ============================================================

observed_r6_log2fc = {

    dinucleotide:
        dinucleotide_log2fc_observed[
            "R6"
        ][
            dinucleotide
        ]

    for dinucleotide in dinucleotides
}


figure_2_14b = pd.DataFrame({

    "Dinucleotide":
        dinucleotide_order,

    "Observed_R6_vs_R1_log2FC":
        [
            observed_r6_log2fc[d]
            for d in dinucleotide_order
        ]
})


# ============================================================
# 13. Figure 2.14c
#
# Enrichment relative to GC-matched expectation
# ============================================================

#
# Figure 2.13 gives:
#
#     observed R6/R1 log2 fold-change
#
# and:
#
#     expected R6/R1 log2 fold-change
#     from GC-matched random sequences.
#
#
# Therefore enrichment relative to the GC-matched
# expectation is:
#
#     observed log2FC - random log2FC
#
# which is equivalent to:
#
# log2(
#
#     observed R6/R1 fold-change
#     ---------------------------
#     expected R6/R1 fold-change
#
# )
#

gc_matched_relative_enrichment = {}


for dinucleotide in dinucleotides:

    observed = (
        dinucleotide_log2fc_observed[
            "R6"
        ][
            dinucleotide
        ]
    )


    expected = (
        dinucleotide_log2fc_random_mean[
            "R6"
        ][
            dinucleotide
        ]
    )


    gc_matched_relative_enrichment[
        dinucleotide
    ] = (
        observed
        -
        expected
    )


figure_2_14c = pd.DataFrame({

    "Dinucleotide":
        dinucleotide_order,

    "Observed_R6_vs_R1_log2FC":
        [
            observed_r6_log2fc[d]
            for d in dinucleotide_order
        ],

    "GC_matched_expected_R6_vs_R1_log2FC":
        [
            dinucleotide_log2fc_random_mean[
                "R6"
            ][d]

            for d in dinucleotide_order
        ],

    "Enrichment_relative_to_GC_matched":
        [
            gc_matched_relative_enrichment[d]
            for d in dinucleotide_order
        ]
})


# ============================================================
# 14. Combined Figure 2.14 table
# ============================================================

figure_2_14 = pd.DataFrame({

    "Dinucleotide":
        dinucleotide_order,

    "Mean_cyclisability":
        [
            dinucleotide_mean_cyclisability[d]
            for d in dinucleotide_order
        ],

    "Mechanical_effect":
        [
            dinucleotide_mechanical_effect[d]
            for d in dinucleotide_order
        ],

    "Observed_R6_vs_R1_log2FC":
        [
            dinucleotide_log2fc_observed[
                "R6"
            ][d]

            for d in dinucleotide_order
        ],

    "GC_matched_expected_R6_vs_R1_log2FC":
        [
            dinucleotide_log2fc_random_mean[
                "R6"
            ][d]

            for d in dinucleotide_order
        ],

    "Enrichment_relative_to_GC_matched":
        [
            gc_matched_relative_enrichment[d]
            for d in dinucleotide_order
        ]
})


print(
    "\nFigure 2.14 data:"
)

print(
    figure_2_14
)


# ============================================================
# FIGURE OUTPUT SUMMARY
# ============================================================

# Figure 2.14a:
#
#     figure_2_14a["Dinucleotide"]
#     figure_2_14a["Mechanical_effect"]
#
#
# Figure 2.14b:
#
#     figure_2_14b["Dinucleotide"]
#     figure_2_14b["Observed_R6_vs_R1_log2FC"]
#
#
# Figure 2.14c:
#
#     figure_2_14c["Dinucleotide"]
#     figure_2_14c["Enrichment_relative_to_GC_matched"]
#
#
# All three panels use:
#
#     dinucleotide_order
#
# so that the dinucleotides are ordered according to
# mechanical effect as described in the thesis.