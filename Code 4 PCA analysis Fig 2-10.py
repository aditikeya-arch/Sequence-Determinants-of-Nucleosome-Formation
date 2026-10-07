

"""
PCA of intrinsic cyclisability profiles — Figure 2.10
======================================================

PURPOSE
-------
Generate all data required for Figure 2.10.

Starting from R6.csv:

1. Select the 10,000 sequences with highest CPM.
2. Collapse near-duplicate sequences differing at <10 positions,
   retaining the higher-CPM sequence.
3. Calculate intrinsic cyclisability profiles.
4. Mean-centre each sequence's profile individually.
5. Perform PCA with 10 components.
6. Calculate PC1/PC2 deciles and their mean ORIGINAL
   cyclisability profiles.
7. Generate synthetic negative-control profiles preserving
   position-dependent mean and standard deviation.
8. Mean-centre synthetic profiles and perform PCA identically.

EXPECTED RESULT
---------------
The near-duplicate filtering should leave 9,803 sequences.

REQUIRED FUNCTIONS
------------------
load_model
pred

INPUT
-----
R6.csv

with columns:

    Sequence
    Counts Per Million

OUTPUT VARIABLES
----------------
top10000
selected
c0_mat
X
pca
Y
explained_variance
cumulative_variance
pc_components
pc1_decile_profiles
pc2_decile_profiles

synthetic_profiles
synthetic_X
synthetic_pca
synthetic_Y
synthetic_explained_variance
synthetic_cumulative_variance
synthetic_components
"""

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA


# ============================================================
# 1. Load Round 6
# ============================================================

r6 = pd.read_csv("R6.csv")

r6 = r6.sort_values(
    "Counts Per Million",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 2. Select top 10,000 sequences
# ============================================================

top10000 = r6.iloc[:10000].copy()


# ============================================================
# 3. Remove near-duplicate sequences
#
# Sequences are already ordered highest -> lowest CPM.
#
# Therefore, when two sequences differ at fewer than 10
# positions, the lower-CPM sequence is discarded.
#
# Because we work down the CPM-ranked list, the first sequence
# encountered is always the one retained.
# ============================================================

def hamming_distance(seq1, seq2):
    """
    Number of positions at which two equal-length
    sequences differ.
    """

    if len(seq1) != len(seq2):
        raise ValueError(
            "Sequences must have equal length for Hamming distance."
        )

    return sum(a != b for a, b in zip(seq1, seq2))


kept_indices = []
kept_sequences = []


for idx, row in top10000.iterrows():

    candidate = row["Sequence"]

    keep = True

    for existing in kept_sequences:

        if hamming_distance(candidate, existing) < 10:
            keep = False
            break

    if keep:
        kept_indices.append(idx)
        kept_sequences.append(candidate)


selected = (
    top10000
    .loc[kept_indices]
    .reset_index(drop=True)
)


print("Top sequences before filtering:", len(top10000))
print("Sequences after filtering:", len(selected))


# Sanity check against thesis result
assert len(selected) == 9803, (
    f"Expected 9803 sequences, obtained {len(selected)}. "
    "Check the near-duplicate filtering criterion."
)


# ============================================================
# 4. Calculate intrinsic cyclisability profiles
# ============================================================

seq = selected["Sequence"].astype(str).tolist()

L = len(seq[0])

if not all(len(s) == L for s in seq):
    raise ValueError("All sequences must have the same length.")


seqparsed = [
    s[i:i+50]
    for s in seq
    for i in range(L - 50 + 1)
]


model = load_model(0)

c0_vals = np.array(
    pred(model, seqparsed)
)

c0_mat = c0_vals.reshape(
    len(seq),
    L - 50 + 1
)


# For 147-bp sequences:
#
# c0_mat.shape == (9803, 98)

print("Cyclisability matrix shape:", c0_mat.shape)


# ============================================================
# 5. Mean-centre EACH PROFILE individually
#
# This removes differences in overall flexibility and leaves
# variation in the SHAPE of the cyclisability profile.
# ============================================================

profile_means = np.mean(
    c0_mat,
    axis=1,
    keepdims=True
)

X = c0_mat - profile_means


# Check mean-centering
assert np.allclose(
    np.mean(X, axis=1),
    0
)


# ============================================================
# 6. PCA
# ============================================================

pca = PCA(n_components=10)

Y = pca.fit_transform(X)


# Y has shape:
#
#     9803 x 10
#
# Columns correspond to PC1, PC2, ... PC10.


# ============================================================
# Figure 2.10a
#
# Fraction of variance explained by each PC
# and cumulative variance explained.
# ============================================================

explained_variance = pca.explained_variance_ratio_

cumulative_variance = np.cumsum(
    explained_variance
)


# ============================================================
# Figure 2.10b
#
# Principal component shapes.
#
# pc_components[0] = PC1
# pc_components[1] = PC2
# ============================================================

pc_components = pca.components_


# Position assigned to centre of each 50-bp window.
#
# Thesis convention:
# first window -> position 25
# second window -> position 26
# etc.

c0_positions = np.arange(
    25,
    25 + c0_mat.shape[1]
)


# ============================================================
# 7. Add PC scores to sequence table
# ============================================================

for i in range(10):

    selected[f"PC{i+1}"] = Y[:, i]


# ============================================================
# 8. Divide sequences into deciles by PC score
#
# Figure 2.10c
# ============================================================

selected["PC1_decile"] = pd.qcut(
    selected["PC1"],
    q=10,
    labels=False
)

selected["PC2_decile"] = pd.qcut(
    selected["PC2"],
    q=10,
    labels=False
)


# ============================================================
# Mean ORIGINAL cyclisability profile within each decile
#
# Note:
# We average c0_mat here, NOT the mean-centred X matrix.
#
# This gives the actual mean intrinsic cyclisability profiles
# displayed for the different PC groups.
# ============================================================

pc1_decile_profiles = np.zeros(
    (10, c0_mat.shape[1])
)

pc2_decile_profiles = np.zeros(
    (10, c0_mat.shape[1])
)


for decile in range(10):

    pc1_mask = (
        selected["PC1_decile"].to_numpy()
        == decile
    )

    pc2_mask = (
        selected["PC2_decile"].to_numpy()
        == decile
    )


    pc1_decile_profiles[decile] = np.mean(
        c0_mat[pc1_mask],
        axis=0
    )

    pc2_decile_profiles[decile] = np.mean(
        c0_mat[pc2_mask],
        axis=0
    )


# ============================================================
# Figure 2.10d
#
# Coordinates of every sequence in PC1-PC2 space.
# ============================================================

pc1_scores = Y[:, 0]
pc2_scores = Y[:, 1]


# ============================================================
# 9. SYNTHETIC NEGATIVE CONTROL
#
# Figure 2.10e-g
#
# At every position:
#
#   mean = observed mean at that position
#   SD   = observed SD at that position
#
# Values at different positions are drawn independently,
# destroying position-to-position correlations.
# ============================================================

position_mean = np.mean(
    c0_mat,
    axis=0
)

position_std = np.std(
    c0_mat,
    axis=0
)


# Reproducible random number generator
rng = np.random.default_rng(1)


synthetic_profiles = rng.normal(
    loc=position_mean,
    scale=position_std,
    size=c0_mat.shape
)


# ============================================================
# Figure 2.10e data
#
# Check that synthetic data reproduce the position-dependent
# mean and SD of the real data.
# ============================================================

synthetic_position_mean = np.mean(
    synthetic_profiles,
    axis=0
)

synthetic_position_std = np.std(
    synthetic_profiles,
    axis=0
)


# ============================================================
# 10. Mean-centre every synthetic profile
#
# Same treatment as the real data.
# ============================================================

synthetic_profile_means = np.mean(
    synthetic_profiles,
    axis=1,
    keepdims=True
)

synthetic_X = (
    synthetic_profiles
    - synthetic_profile_means
)


# ============================================================
# 11. PCA of synthetic control
# ============================================================

synthetic_pca = PCA(
    n_components=10
)

synthetic_Y = synthetic_pca.fit_transform(
    synthetic_X
)


# ============================================================
# Figure 2.10f
#
# Variance explained by PCs of synthetic dataset
# ============================================================

synthetic_explained_variance = (
    synthetic_pca.explained_variance_ratio_
)

synthetic_cumulative_variance = np.cumsum(
    synthetic_explained_variance
)


# ============================================================
# Figure 2.10g
#
# Principal component shapes of synthetic dataset
# ============================================================

synthetic_components = (
    synthetic_pca.components_
)


# ============================================================
# Useful numerical summary
# ============================================================

pca_summary = pd.DataFrame({

    "PC": np.arange(1, 11),

    "Real variance explained":
        explained_variance,

    "Real cumulative variance":
        cumulative_variance,

    "Synthetic variance explained":
        synthetic_explained_variance,

    "Synthetic cumulative variance":
        synthetic_cumulative_variance
})


print(pca_summary)