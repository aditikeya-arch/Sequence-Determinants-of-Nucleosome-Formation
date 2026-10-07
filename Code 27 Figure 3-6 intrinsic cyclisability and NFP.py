

"""
Code 27 Figure 3-6 intrinsic cyclisability and NFP.py
=====================================================

WHAT THIS SCRIPT DOES
---------------------
Generates the data underlying both panels of Figure 3.6.

For each of the 2,000 sequences in the competitive
nucleosome reconstitution library, the mean predicted
intrinsic cyclisability is calculated from its complete
98-point intrinsic-cyclisability profile.

LEFT PANEL
----------
Compares mean intrinsic cyclisability with experimentally
measured nucleosome formation propensity (NFP) for all
2,000 sequences.

The 200 SELEX sequences and Widom 601 are identified
separately in the output so that they can be highlighted
when plotting.

A Pearson correlation is calculated using all 2,000
sequences.


RIGHT PANEL
-----------
Performs the same comparison for the 200 designed
G1-G10 sequences only.

These occupy library positions 601-800 and consist of
10 groups of 20 sequences each.

A Pearson correlation is calculated using these 200
sequences.


VARIABLES THAT MUST ALREADY EXIST
---------------------------------
seq

    Ordered list containing the 2,000 experimental
    147-bp sequences.

NFP

    Array containing the experimentally measured
    nucleosome formation propensity of the same 2,000
    sequences, in the same order.

    This is calculated in:

    Code 26 Figure 3-5 nucleosome formation propensity
    by sequence group.py


FUNCTIONS THAT MUST ALREADY EXIST
---------------------------------
load_model

pred

These are the intrinsic-cyclisability prediction
functions used throughout the thesis.


OPTIONAL VARIABLE
-----------------
c0_mat

    If already available, this should be a 2000 x 98
    matrix containing the intrinsic-cyclisability
    profiles of all 2,000 sequences.

    If c0_mat does not already exist, it is calculated
    in this script.


OUTPUTS
-------
figure_3_6_left

    Data underlying the left panel.

    Columns:

        Library_position
        Sequence
        Mean_intrinsic_cyclisability
        NFP
        Plot_class


figure_3_6_left_correlation

    Pearson r and p value for all 2,000 sequences.


figure_3_6_right

    Data underlying the right panel.

    Contains the 200 G1-G10 sequences.

    Columns:

        Library_position
        Sequence
        Group
        Mean_intrinsic_cyclisability
        NFP


figure_3_6_right_correlation

    Pearson r and p value for the 200 G1-G10 sequences.


OTHER SCRIPTS THAT MUST BE RUN FIRST
------------------------------------
Code 26 Figure 3-5 nucleosome formation propensity
by sequence group.py

Alternatively, seq and NFP can be supplied directly.


NOTES
-----
Mean intrinsic cyclisability is the arithmetic mean of
the 98 predictions obtained from overlapping 50-bp
windows across each 147-bp sequence.

No weighting is used.

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


# ============================================================
# 2. Check required inputs
# ============================================================

if "seq" not in globals():

    raise RuntimeError(
        "seq must already exist and contain the 2,000 "
        "experimental sequences in library order."
    )


if "NFP" not in globals():

    raise RuntimeError(
        "NFP must already exist and contain the measured "
        "nucleosome formation propensity of all 2,000 "
        "sequences."
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


NFP = np.asarray(
    NFP,
    dtype=float
)


assert len(seq) == N_SEQUENCES
assert len(NFP) == N_SEQUENCES


assert all(
    len(sequence) == SEQUENCE_LENGTH
    for sequence in seq
)


# ============================================================
# 3. Calculate intrinsic-cyclisability profiles
# ============================================================

# Reuse c0_mat if it already exists and has the expected
# dimensions. Otherwise calculate it here.

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
# 4. Mean intrinsic cyclisability of each sequence
# ============================================================

# This follows the original analysis, where each
# 147-bp sequence is represented by the mean of its
# complete 98-point intrinsic-cyclisability profile.

c0_mean = np.nanmean(
    c0_mat,
    axis=1
)


assert len(c0_mean) == 2000


# ============================================================
# 5. LEFT PANEL
#    All 2,000 sequences
# ============================================================

# The original figure distinguishes:
#
#     SELEX sequences     positions 1-200
#     Widom 601           position 801
#     all other sequences


plot_class = np.full(
    N_SEQUENCES,
    "Other",
    dtype=object
)


plot_class[0:200] = "SELEX"

plot_class[800] = "Widom 601"


figure_3_6_left = pd.DataFrame({

    "Library_position":
        np.arange(
            1,
            2001
        ),

    "Sequence":
        seq,

    "Mean_intrinsic_cyclisability":
        c0_mean,

    "NFP":
        NFP,

    "Plot_class":
        plot_class
})


# ============================================================
# 6. Pearson correlation for the left panel
# ============================================================

valid_left = (

    np.isfinite(c0_mean)

    &

    np.isfinite(NFP)
)


r_left, p_left = pearsonr(

    c0_mean[
        valid_left
    ],

    NFP[
        valid_left
    ]
)


figure_3_6_left_correlation = pd.DataFrame({

    "N": [
        int(
            valid_left.sum()
        )
    ],

    "Pearson_r": [
        r_left
    ],

    "P_value": [
        p_left
    ]
})


# ============================================================
# 7. RIGHT PANEL
#    G1-G10 sequences only
# ============================================================

# Library positions 601-800 correspond to Python
# indices 600:800.

G_START = 600
G_END = 800

N_GROUPS = 10
SEQUENCES_PER_GROUP = 20


group_labels = []


for group_number in range(
    1,
    N_GROUPS + 1
):

    group_labels += [

        f"G{group_number}"

    ] * SEQUENCES_PER_GROUP


assert len(group_labels) == 200


figure_3_6_right = pd.DataFrame({

    "Library_position":
        np.arange(
            601,
            801
        ),

    "Sequence":
        seq[
            G_START:G_END
        ],

    "Group":
        group_labels,

    "Mean_intrinsic_cyclisability":
        c0_mean[
            G_START:G_END
        ],

    "NFP":
        NFP[
            G_START:G_END
        ]
})


# ============================================================
# 8. Pearson correlation for the right panel
# ============================================================

x_right = figure_3_6_right[
    "Mean_intrinsic_cyclisability"
].to_numpy()


y_right = figure_3_6_right[
    "NFP"
].to_numpy()


valid_right = (

    np.isfinite(x_right)

    &

    np.isfinite(y_right)
)


r_right, p_right = pearsonr(

    x_right[
        valid_right
    ],

    y_right[
        valid_right
    ]
)


figure_3_6_right_correlation = pd.DataFrame({

    "N": [
        int(
            valid_right.sum()
        )
    ],

    "Pearson_r": [
        r_right
    ],

    "P_value": [
        p_right
    ]
})


# ============================================================
# 9. Optional G1-G10 group summary
# ============================================================

# This is not required for the scatter plot itself, but is
# useful for checking the progression across the ten
# designed groups.

figure_3_6_G_summary = (

    figure_3_6_right

    .groupby(
        "Group",
        sort=False
    )

    .agg(

        N_sequences=(
            "Sequence",
            "count"
        ),

        Mean_intrinsic_cyclisability=(
            "Mean_intrinsic_cyclisability",
            "mean"
        ),

        SD_intrinsic_cyclisability=(
            "Mean_intrinsic_cyclisability",
            "std"
        ),

        Mean_NFP=(
            "NFP",
            "mean"
        ),

        SD_NFP=(
            "NFP",
            "std"
        )
    )

    .reset_index()
)


# ============================================================
# 10. Final checks
# ============================================================

assert len(
    figure_3_6_left
) == 2000


assert (
    figure_3_6_left[
        "Plot_class"
    ]
    .value_counts()
    .to_dict()
    ==
    {
        "Other": 1799,
        "SELEX": 200,
        "Widom 601": 1
    }
)


assert len(
    figure_3_6_right
) == 200


assert (
    figure_3_6_right[
        "Group"
    ]
    .value_counts()
    .eq(20)
    .all()
)


# ============================================================
# 11. Report correlations
# ============================================================

print(
    "Figure 3.6 left panel"
)

print(
    f"N = {valid_left.sum()}"
)

print(
    f"Pearson r = {r_left:.4f}"
)

print(
    f"P = {p_left:.4g}"
)


print(
    "\nFigure 3.6 right panel"
)

print(
    f"N = {valid_right.sum()}"
)

print(
    f"Pearson r = {r_right:.4f}"
)

print(
    f"P = {p_right:.4g}"
)


# ============================================================
# OUTPUT SUMMARY
# ============================================================

# Mean intrinsic cyclisability of every sequence:
#
#     c0_mean
#
#
# LEFT PANEL
# ----------
#
# Scatter-plot data:
#
#     figure_3_6_left
#
# Correlation:
#
#     figure_3_6_left_correlation
#
#
# RIGHT PANEL
# -----------
#
# G1-G10 scatter-plot data:
#
#     figure_3_6_right
#
# Correlation:
#
#     figure_3_6_right_correlation
#
#
# Additional G1-G10 group summary:
#
#     figure_3_6_G_summary
#
#
# No plotting is performed.