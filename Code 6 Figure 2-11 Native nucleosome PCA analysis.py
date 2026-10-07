

"""
Code 6 - Figure 2-11 Native nucleosome cyclisability and PCA
=============================================================

PURPOSE
-------
Generate the data used for Figure 2.11.

This script:

1. Calculates intrinsic cyclisability profiles for all native
   S. cerevisiae nucleosomal sequences.

2. Calculates mean intrinsic cyclisability profiles for:
       - all native nucleosomes
       - top 5% of native nucleosomes by Widom NCP score
       - bottom 5% of native nucleosomes by Widom NCP score

3. Selects the 10,000 most enriched sequences from round 6
   of SELEX and removes near-duplicate sequences.

   Sequences differing at fewer than 10 positions are collapsed,
   retaining the sequence with the higher CPM.

   For the R6 dataset used in the thesis, this leaves 9,803
   sequences.

4. Calculates the mean intrinsic cyclisability profile of this
   filtered high-enrichment R6 set for comparison with native
   nucleosomes.

5. Performs PCA on the intrinsic cyclisability profiles of all
   native nucleosomes.

6. Calculates:
       - variance explained by each PC
       - cumulative variance explained
       - PC1 and PC2 coefficients
       - PC1 and PC2 scores for each native nucleosome

7. Divides native nucleosomes into deciles according to PC1
   and PC2 score and calculates the mean intrinsic
   cyclisability profile within each decile.


THESIS FIGURE
-------------
Figure 2.11

(a)
Mean intrinsic cyclisability profiles for native yeast
nucleosomal DNA.

Left:
    all native nucleosomes
    top 5% by Widom NCP score
    bottom 5% by Widom NCP score

Right:
    comparison with the highly enriched R6 SELEX sequences.

The figure caption refers to the "top 10,000" R6 sequences.
Throughout this analysis, this means the 10,000 sequences
selected by CPM before redundancy filtering. Removal of
near-duplicates leaves 9,803 sequences.

(b)
Fraction of variance explained by each principal component
and cumulative variance explained.

(c)
First two principal components as functions of position.

(d)
Mean intrinsic cyclisability profiles of native nucleosomes
grouped into deciles according to PC1 or PC2 score.


REQUIRED PREVIOUS SCRIPT
------------------------
Code 5 Native yeast nucleosomal sequences.py

This must have been run before this script.


REQUIRED VARIABLES IN MEMORY
----------------------------
native_nucleosomes : pandas.DataFrame

    Produced by Code 5.

    Required columns:

        Chromosome
        Dyad
        NCP_score
        Sequence

    Sequence must contain the 147-bp SacCer3 DNA sequence
    centred on each native nucleosome dyad.


REQUIRED FILE
-------------
R6.csv

    Round 6 SELEX data.

    Required columns:

        Sequence
        Counts Per Million


REQUIRED FUNCTIONS
------------------
load_model
pred

These functions are supplied by the Cyclizability Prediction code:

https://github.com/codergirl1106/Cyclizability-Prediction-Website/


OUTPUT VARIABLES
----------------
native_c0_mat

all_native_mean_profile
top5_mean_profile
bottom5_mean_profile

r6_top10000
r6_top_filtered
selex_top_filtered_c0_mat
selex_top_filtered_mean_profile

native_X
native_pca
native_Y

native_explained_variance
native_cumulative_variance

native_pc_components
native_pc1
native_pc2

native_pc1_scores
native_pc2_scores

native_pc1_decile_profiles
native_pc2_decile_profiles

c0_positions
"""


import numpy as np
import pandas as pd

from sklearn.decomposition import PCA


# ============================================================
# 1. Load intrinsic cyclisability model
# ============================================================

model = load_model(0)


# ============================================================
# 2. Function for calculating intrinsic cyclisability profiles
# ============================================================

def calculate_c0_profiles(sequences):

    sequences = list(sequences)

    L = len(sequences[0])

    if not all(len(s) == L for s in sequences):

        raise ValueError(
            "All sequences must have the same length."
        )


    # Divide every sequence into overlapping 50-bp windows.
    #
    # For a 147-bp sequence:
    #
    #     147 - 50 + 1 = 98
    #
    # predictions are obtained.

    seqparsed = [

        s[i:i+50]

        for s in sequences

        for i in range(L - 50 + 1)
    ]


    # Predict intrinsic cyclisability for every window.

    c0_vals = np.array(

        pred(
            model,
            seqparsed
        )
    )


    # Reshape predictions so that:
    #
    #     rows    = sequences
    #     columns = positions along the sequence
    #
    # For 147-bp sequences:
    #
    #     shape = N x 98

    c0_mat = c0_vals.reshape(

        len(sequences),

        L - 50 + 1
    )


    return c0_mat


# ============================================================
# PART A
#
# Native yeast nucleosome intrinsic cyclisability
# ============================================================


# ============================================================
# 3. Obtain native nucleosomal sequences
# ============================================================

native_sequences = (

    native_nucleosomes[
        "Sequence"
    ]

    .astype(str)

    .tolist()
)


native_ncp_scores = (

    native_nucleosomes[
        "NCP_score"
    ]

    .to_numpy(dtype=float)
)


print(
    "Number of native nucleosomes:",
    len(native_sequences)
)


# The native nucleosomal DNA sequences used here should all
# contain exactly 147 bp.

native_sequence_lengths = np.array([

    len(s)

    for s in native_sequences
])


if not np.all(
    native_sequence_lengths == 147
):

    raise ValueError(
        "Not all native nucleosome sequences are 147 bp."
    )


# ============================================================
# 4. Calculate intrinsic cyclisability profiles
# ============================================================

native_c0_mat = calculate_c0_profiles(
    native_sequences
)


print(
    "Native cyclisability matrix shape:",
    native_c0_mat.shape
)


# For approximately 67,000 native nucleosomes the expected
# shape is approximately:
#
#     67000 x 98


# ============================================================
# 5. Position coordinates
# ============================================================

#
# The first cyclisability prediction corresponds to the first
# 50-bp window.
#
# Following the convention used elsewhere in Chapter 2, this
# prediction is assigned nominally to position 25.
#
# Therefore the 98 predictions for a 147-bp sequence correspond
# to positions:
#
#     25, 26, ..., 122
#

c0_positions = np.arange(

    25,

    25 + native_c0_mat.shape[1]
)


# ============================================================
# FIGURE 2.11a LEFT
# ============================================================


# ============================================================
# 6. Mean profile of ALL native nucleosomes
# ============================================================

all_native_mean_profile = np.mean(

    native_c0_mat,

    axis=0
)


# ============================================================
# 7. Rank native nucleosomes by Widom NCP score
# ============================================================

n_native = len(
    native_nucleosomes
)


# Number corresponding to 5% of the native nucleosomes.

n_5percent = int(
    np.ceil(
        0.05 * n_native
    )
)


# Sort indices from lowest NCP score to highest NCP score.

ncp_order = np.argsort(
    native_ncp_scores
)


# Lowest 5%.

bottom5_indices = ncp_order[
    :n_5percent
]


# Highest 5%.

top5_indices = ncp_order[
    -n_5percent:
]


# ============================================================
# 8. Mean profiles of top and bottom 5%
# ============================================================

bottom5_mean_profile = np.mean(

    native_c0_mat[
        bottom5_indices
    ],

    axis=0
)


top5_mean_profile = np.mean(

    native_c0_mat[
        top5_indices
    ],

    axis=0
)


print(
    "Number of nucleosomes in bottom 5%:",
    len(bottom5_indices)
)


print(
    "Number of nucleosomes in top 5%:",
    len(top5_indices)
)


# Store these classifications with the nucleosome information.

native_nucleosomes["NCP_group"] = "middle"


native_nucleosomes.loc[
    bottom5_indices,
    "NCP_group"
] = "bottom_5_percent"


native_nucleosomes.loc[
    top5_indices,
    "NCP_group"
] = "top_5_percent"


# ============================================================
# FIGURE 2.11a RIGHT
#
# Comparison with highly enriched R6 SELEX sequences
# ============================================================


# ============================================================
# 9. Read R6 SELEX data
# ============================================================

r6 = pd.read_csv(
    "R6.csv"
)


# Sort from highest to lowest CPM.

r6 = (

    r6

    .sort_values(
        "Counts Per Million",
        ascending=False
    )

    .reset_index(drop=True)
)


# ============================================================
# 10. Select top 10,000 sequences BEFORE redundancy filtering
# ============================================================

r6_top10000 = (

    r6

    .iloc[:10000]

    .copy()
)


print(
    "R6 sequences before redundancy filtering:",
    len(r6_top10000)
)


# ============================================================
# 11. Hamming distance
# ============================================================

def hamming_distance(seq1, seq2):

    if len(seq1) != len(seq2):

        raise ValueError(
            "Sequences must have equal length "
            "for Hamming-distance comparison."
        )


    return sum(

        a != b

        for a, b in zip(
            seq1,
            seq2
        )
    )


# ============================================================
# 12. Remove near-duplicate R6 sequences
# ============================================================

#
# r6_top10000 is already ordered from highest CPM to lowest CPM.
#
# We proceed through that list in order.
#
# A candidate sequence is retained only if it differs from
# every previously retained sequence at 10 or more positions.
#
# Therefore, if two sequences differ at fewer than 10
# positions, the higher-CPM sequence is retained.
#

kept_indices = []

kept_sequences = []


for idx, row in r6_top10000.iterrows():

    candidate = str(
        row["Sequence"]
    )


    keep = True


    for existing in kept_sequences:

        if hamming_distance(
            candidate,
            existing
        ) < 10:

            keep = False

            break


    if keep:

        kept_indices.append(
            idx
        )

        kept_sequences.append(
            candidate
        )


# Construct the filtered dataset.

r6_top_filtered = (

    r6_top10000

    .loc[
        kept_indices
    ]

    .reset_index(drop=True)
)


print(
    "R6 sequences after redundancy filtering:",
    len(r6_top_filtered)
)


# For the dataset used in the thesis this filtering should
# reduce the top 10,000 sequences to 9,803.

assert len(r6_top_filtered) == 9803, (

    "Expected 9,803 sequences after filtering, but obtained "
    f"{len(r6_top_filtered)}. "

    "Check the R6 input data and the redundancy-filtering "
    "criterion."
)


# ============================================================
# 13. Calculate cyclisability of filtered high-enrichment R6 set
# ============================================================

selex_top_filtered_sequences = (

    r6_top_filtered[
        "Sequence"
    ]

    .astype(str)

    .tolist()
)


selex_top_filtered_c0_mat = (

    calculate_c0_profiles(
        selex_top_filtered_sequences
    )
)


# ============================================================
# 14. Mean R6 cyclisability profile
# ============================================================

#
# Each of the 9,803 retained sequences contributes once.
#
# The profile is therefore NOT weighted by CPM.
#

selex_top_filtered_mean_profile = np.mean(

    selex_top_filtered_c0_mat,

    axis=0
)


# ============================================================
# PART B
#
# PCA of native nucleosomal cyclisability profiles
# ============================================================


# ============================================================
# 15. Mean-centre EACH native nucleosome profile
# ============================================================

#
# This follows the same PCA procedure used for Figure 2.10.
#
# Each sequence's own mean intrinsic cyclisability is
# subtracted from all 98 positions.
#
# PCA therefore describes variation in the SHAPE of the
# cyclisability profile rather than variation in the overall
# mean cyclisability of different sequences.
#

native_profile_means = np.mean(

    native_c0_mat,

    axis=1,

    keepdims=True
)


native_X = (

    native_c0_mat

    - native_profile_means
)


# Confirm that every mean-centred profile has mean zero
# (within numerical precision).

assert np.allclose(

    np.mean(
        native_X,
        axis=1
    ),

    0
)


# ============================================================
# 16. PCA
# ============================================================

native_pca = PCA(
    n_components=10
)


native_Y = native_pca.fit_transform(
    native_X
)


# Dimensions:
#
# native_X
#
#     number of native nucleosomes x 98
#
#
# native_Y
#
#     number of native nucleosomes x 10


# ============================================================
# FIGURE 2.11b
#
# Variance explained by principal components
# ============================================================

native_explained_variance = (

    native_pca
    .explained_variance_ratio_
)


native_cumulative_variance = np.cumsum(

    native_explained_variance
)


# Convenient table containing the panel-b data.

native_pca_summary = pd.DataFrame({

    "PC":
        np.arange(1, 11),

    "Variance_explained":
        native_explained_variance,

    "Cumulative_variance":
        native_cumulative_variance
})


print(
    native_pca_summary
)


# ============================================================
# FIGURE 2.11c
#
# PC1 and PC2 as functions of position
# ============================================================

native_pc_components = (
    native_pca.components_
)


# Each row contains the 98 coefficients defining one PC.

native_pc1 = (
    native_pc_components[0]
)


native_pc2 = (
    native_pc_components[1]
)


# Plot against:
#
#     c0_positions


# ============================================================
# FIGURE 2.11d
#
# Native nucleosome profiles grouped by PC score
# ============================================================


# ============================================================
# 17. Obtain PC1 and PC2 scores for every nucleosome
# ============================================================

native_pc1_scores = (
    native_Y[:, 0]
)


native_pc2_scores = (
    native_Y[:, 1]
)


# Store scores alongside nucleosome information.

native_nucleosomes["PC1"] = (
    native_pc1_scores
)


native_nucleosomes["PC2"] = (
    native_pc2_scores
)


# ============================================================
# 18. Divide PC1 and PC2 scores into deciles
# ============================================================

native_nucleosomes["PC1_decile"] = pd.qcut(

    native_nucleosomes[
        "PC1"
    ],

    q=10,

    labels=False
)


native_nucleosomes["PC2_decile"] = pd.qcut(

    native_nucleosomes[
        "PC2"
    ],

    q=10,

    labels=False
)


# Deciles are numbered:
#
#     0, 1, 2, ..., 9
#
# where:
#
#     0 = lowest PC scores
#     9 = highest PC scores


# ============================================================
# 19. Mean ORIGINAL profiles within PC1 deciles
# ============================================================

#
# IMPORTANT:
#
# The mean-centred profiles (native_X) are used to determine
# the PCA scores and therefore the decile membership.
#
# Once those groups have been defined, the ORIGINAL
# cyclisability profiles (native_c0_mat) are averaged.
#

native_pc1_decile_profiles = np.zeros(

    (
        10,
        native_c0_mat.shape[1]
    )
)


for decile in range(10):

    mask = (

        native_nucleosomes[
            "PC1_decile"
        ]

        .to_numpy()

        == decile
    )


    native_pc1_decile_profiles[
        decile
    ] = np.mean(

        native_c0_mat[
            mask
        ],

        axis=0
    )


# ============================================================
# 20. Mean ORIGINAL profiles within PC2 deciles
# ============================================================

native_pc2_decile_profiles = np.zeros(

    (
        10,
        native_c0_mat.shape[1]
    )
)


for decile in range(10):

    mask = (

        native_nucleosomes[
            "PC2_decile"
        ]

        .to_numpy()

        == decile
    )


    native_pc2_decile_profiles[
        decile
    ] = np.mean(

        native_c0_mat[
            mask
        ],

        axis=0
    )


# ============================================================
# FIGURE 2.11 OUTPUT SUMMARY
# ============================================================


# ------------------------------------------------------------
# FIGURE 2.11a - LEFT
# ------------------------------------------------------------
#
# x:
#
#     c0_positions
#
# y:
#
#     all_native_mean_profile
#     top5_mean_profile
#     bottom5_mean_profile
#


# ------------------------------------------------------------
# FIGURE 2.11a - RIGHT
# ------------------------------------------------------------
#
# x:
#
#     c0_positions
#
# y:
#
#     all_native_mean_profile
#     top5_mean_profile
#     bottom5_mean_profile
#     selex_top_filtered_mean_profile
#
# The SELEX profile represents the 9,803 sequences remaining
# after selecting the top 10,000 R6 sequences by CPM and
# removing near-duplicates.
#


# ------------------------------------------------------------
# FIGURE 2.11b
# ------------------------------------------------------------
#
#     native_explained_variance
#     native_cumulative_variance
#
# or equivalently:
#
#     native_pca_summary
#


# ------------------------------------------------------------
# FIGURE 2.11c
# ------------------------------------------------------------
#
# x:
#
#     c0_positions
#
# y:
#
#     native_pc1
#     native_pc2
#


# ------------------------------------------------------------
# FIGURE 2.11d - LEFT
# ------------------------------------------------------------
#
#     native_pc1_decile_profiles
#
# Shape:
#
#     10 x 98
#


# ------------------------------------------------------------
# FIGURE 2.11d - RIGHT
# ------------------------------------------------------------
#
#     native_pc2_decile_profiles
#
# Shape:
#
#     10 x 98
#