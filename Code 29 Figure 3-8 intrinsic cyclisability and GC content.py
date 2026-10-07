

"""
Code 29 Figure 3-8 intrinsic cyclisability and GC content.py
=============================================================

WHAT THIS SCRIPT DOES
---------------------
Generates the data underlying Figure 3.8.

Figure 3.8 compares mean intrinsic cyclisability with
GC fraction across all 2,000 sequences in the competitive
nucleosome reconstitution library.

For each sequence:

    1. Intrinsic cyclisability is predicted across all
       overlapping 50-bp windows of the 147-bp sequence.

    2. The mean of the resulting 98-point profile is
       calculated.

    3. GC fraction is calculated over the central 113-bp
       variable region, excluding the constant 17-bp
       adapters.

A Pearson correlation is calculated between mean
intrinsic cyclisability and GC fraction.


VARIABLES THAT MUST ALREADY EXIST
---------------------------------
seq

    Ordered list containing the 2,000 experimental
    sequences.

    Every sequence must be 147 bp long.


FUNCTIONS THAT MUST ALREADY EXIST
---------------------------------
load_model

pred

These are the intrinsic-cyclisability prediction
functions used throughout the thesis.


OPTIONAL VARIABLES
------------------
c0_mat

    2000 x 98 matrix containing the intrinsic-
    cyclisability profiles of all sequences.

    If c0_mat does not already exist, it is calculated
    in this script.


OTHER SCRIPTS THAT CAN BE RUN FIRST
-----------------------------------
Code 26 Figure 3-5 nucleosome formation propensity
by sequence group.py

    provides seq

Code 27 Figure 3-6 intrinsic cyclisability and NFP.py

    provides c0_mat


OUTPUTS
-------
c0_mean

    Mean intrinsic cyclisability of each of the
    2,000 sequences.


GC_fraction

    GC fraction of the central 113 bp of each sequence.


figure_3_8

    DataFrame containing:

        Library_position
        Sequence
        Mean_intrinsic_cyclisability
        GC_fraction


figure_3_8_correlation

    Pearson correlation coefficient and p value.


NOTES
-----
GC fraction is calculated over:

    sequence[17:130]

so that the constant 17-bp adapters are excluded.

Mean intrinsic cyclisability is calculated from all
98 overlapping 50-bp windows of the complete 147-bp
experimental sequence.

No plotting is performed.
"""


import numpy as np
import pandas as pd
from scipy.stats import pearsonr


# ============================================================
# 1. Constants
# ============================================================

N_SEQUENCES = 2000

SEQUENCE_LENGTH = 147

WINDOW = 50

PROFILE_LENGTH = (
    SEQUENCE_LENGTH - WINDOW + 1
)  # 98

TRIM_LEFT = 17
TRIM_RIGHT = 17

CENTRAL_LENGTH = 113


# ============================================================
# 2. Check required inputs
# ============================================================

if "seq" not in globals():

    raise RuntimeError(
        "seq must already exist and contain the 2,000 "
        "experimental sequences in library order."
    )


if "load_model" not in globals():

    raise RuntimeError(
        "load_model must already be defined."
    )


if "pred" not in globals():

    raise RuntimeError(
        "pred must already be defined."
    )


seq = [
    str(sequence).upper()
    for sequence in seq
]


assert len(seq) == N_SEQUENCES, (
    f"Expected 2,000 sequences, "
    f"but found {len(seq)}."
)


assert all(
    len(sequence) == SEQUENCE_LENGTH
    for sequence in seq
), "Every sequence must be 147 bp long."


assert all(
    set(sequence).issubset(
        {"A", "C", "G", "T"}
    )
    for sequence in seq
), "Sequences must contain only A, C, G, and T."


# ============================================================
# 3. Calculate intrinsic-cyclisability profiles if required
# ============================================================

if (
    "c0_mat" not in globals()
    or np.asarray(c0_mat).shape != (2000, 98)
):

    model = load_model(0)


    seqparsed = [

        sequence[i:i + WINDOW]

        for sequence in seq

        for i in range(PROFILE_LENGTH)
    ]


    c0_vals = np.asarray(

        pred(
            model,
            seqparsed
        )

    ).reshape(-1)


    c0_mat = c0_vals.reshape(
        N_SEQUENCES,
        PROFILE_LENGTH
    )


c0_mat = np.asarray(
    c0_mat,
    dtype=float
)


assert c0_mat.shape == (
    2000,
    98
)


# ============================================================
# 4. Mean intrinsic cyclisability
# ============================================================

c0_mean = np.nanmean(
    c0_mat,
    axis=1
)


assert len(c0_mean) == 2000


# ============================================================
# 5. Extract the central 113-bp variable region
# ============================================================

central_113 = [

    sequence[
        TRIM_LEFT:
        SEQUENCE_LENGTH - TRIM_RIGHT
    ]

    for sequence in seq
]


assert all(
    len(sequence) == CENTRAL_LENGTH
    for sequence in central_113
)


# ============================================================
# 6. Calculate GC fraction
# ============================================================

def calculate_gc_fraction(sequence):
    """
    Calculate the fraction of bases that are G or C.
    """

    return (

        sequence.count("G")
        + sequence.count("C")

    ) / len(sequence)


GC_fraction = np.asarray([

    calculate_gc_fraction(sequence)

    for sequence in central_113

])


assert len(GC_fraction) == 2000


# ============================================================
# 7. Construct Figure 3.8 data
# ============================================================

figure_3_8 = pd.DataFrame({

    "Library_position":
        np.arange(
            1,
            2001
        ),

    "Sequence":
        seq,

    "Mean_intrinsic_cyclisability":
        c0_mean,

    "GC_fraction":
        GC_fraction
})


# ============================================================
# 8. Pearson correlation
# ============================================================

valid = (

    np.isfinite(
        c0_mean
    )

    &

    np.isfinite(
        GC_fraction
    )
)


r_gc_cyclisability, p_gc_cyclisability = pearsonr(

    c0_mean[
        valid
    ],

    GC_fraction[
        valid
    ]
)


figure_3_8_correlation = pd.DataFrame({

    "N": [
        int(
            valid.sum()
        )
    ],

    "Pearson_r": [
        r_gc_cyclisability
    ],

    "P_value": [
        p_gc_cyclisability
    ]
})


# ============================================================
# 9. Final checks
# ============================================================

assert len(
    figure_3_8
) == 2000


assert (
    (
        figure_3_8[
            "GC_fraction"
        ] >= 0
    )
    &
    (
        figure_3_8[
            "GC_fraction"
        ] <= 1
    )
).all()


# ============================================================
# 10. Report result
# ============================================================

print(
    "Figure 3.8:"
)

print(
    "Mean intrinsic cyclisability versus GC fraction"
)

print(
    f"N = {valid.sum()}"
)

print(
    f"Pearson r = {r_gc_cyclisability:.4f}"
)

print(
    f"P = {p_gc_cyclisability:.4g}"
)


# ============================================================
# OUTPUT SUMMARY
# ============================================================

# Mean intrinsic cyclisability:
#
#     c0_mean
#
#
# Central 113-bp regions:
#
#     central_113
#
#
# GC fractions:
#
#     GC_fraction
#
#
# Data underlying Figure 3.8:
#
#     figure_3_8
#
#
# Pearson correlation:
#
#     figure_3_8_correlation
#
#
# No plotting is performed.