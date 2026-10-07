

"""
Code 26 Figure 3-5 nucleosome formation propensity by sequence group.py
======================================================================

WHAT THIS SCRIPT DOES
---------------------
Generates the data underlying Figure 3.5.

Figure 3.5 compares experimentally measured nucleosome
formation propensity (NFP) across sequence classes in the
2,000-member competitive nucleosome reconstitution library.

NFP is calculated from the sequencing counts in the first
competitive nucleosome reconstitution experiment as:

    ln[(A1 + 0.01) / (A2 Free + 0.01)]

where:

    A1          = nucleosome-associated fraction
    A2 (Free)   = free-DNA fraction

A pseudocount of 0.01 is added before taking the ratio.

The 2,000 sequences are then assigned to their experimental
or design groups according to their position in the library.


INPUT FILE
----------
A1_A6_Counts.csv

Required columns:

    Sequence
    A1
    A2 (Free)


VARIABLES THAT MUST ALREADY EXIST
---------------------------------
None.


FUNCTIONS THAT MUST ALREADY EXIST
---------------------------------
None.


OUTPUTS
-------
NFP

    NumPy array containing the measured NFP of all
    2,000 sequences, in library order.


seq

    Ordered list of the 2,000 experimental sequences.


labels

    Ordered list containing the sequence-class label
    for every sequence.


figure_3_5

    DataFrame containing:

        Library_position
        Sequence
        Label
        A1
        A2 (Free)
        NFP


figure_3_5_summary

    Group-level summary containing:

        Label
        N_sequences
        Mean_NFP
        Median_NFP
        SD_NFP


LIBRARY ORDER
-------------
Positions 1-200
    SELEX

Positions 201-400
    Random

Positions 401-600
    Yeast

Positions 601-800
    G1-G10, 20 sequences per group

Position 801
    Widom 601

Positions 802-901
    Yeast

Positions 902-1001
    Mimic yeast (7)

Positions 1002-1021
    Gaussian flex

Positions 1022-1041
    Native 510 mimic

Positions 1042-1061
    Native 545 mimic

Positions 1062-1171
    Codon-randomized 345

Position 1172
    Native 345

Positions 1173-2000
    Random


NOTES
-----
The final blank row present in A1_A6_Counts.csv is removed
by requiring Sequence to be non-missing.

The sequence order is important because the experimental
classes are defined by position in the 2,000-member library.

No plotting is performed.
"""


import numpy as np
import pandas as pd


# ============================================================
# 1. Load experimental counts
# ============================================================

df = pd.read_csv(
    "A1_A6_Counts.csv"
)


required_columns = [
    "Sequence",
    "A1",
    "A2 (Free)"
]


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column '{column}' "
            "was not found in A1_A6_Counts.csv."
        )


# ============================================================
# 2. Remove the final blank row
# ============================================================

df = df[
    df["Sequence"].notna()
].copy()


df.reset_index(
    drop=True,
    inplace=True
)


assert len(df) == 2000, (
    f"Expected 2,000 sequences, "
    f"but found {len(df)}."
)


# ============================================================
# 3. Check the sequences
# ============================================================

df["Sequence"] = (
    df["Sequence"]
    .astype(str)
    .str.upper()
)


assert df["Sequence"].nunique() == 2000, (
    "The 2,000 library sequences are expected "
    "to be unique."
)


assert (
    df["Sequence"].str.len() == 147
).all(), (
    "Every library sequence must be 147 bp long."
)


assert df["Sequence"].apply(
    lambda s:
    set(s).issubset(
        {"A", "C", "G", "T"}
    )
).all(), (
    "Sequences must contain only A, C, G, and T."
)


# Ordered sequence list used by subsequent analyses.

seq = df[
    "Sequence"
].tolist()


# ============================================================
# 4. Calculate nucleosome formation propensity
# ============================================================

PSEUDOCOUNT = 0.01


NFP = np.log(

    (
        df["A1"].to_numpy(
            dtype=float
        )
        + PSEUDOCOUNT
    )

    /

    (
        df["A2 (Free)"].to_numpy(
            dtype=float
        )
        + PSEUDOCOUNT
    )
)


assert len(NFP) == 2000


# Store NFP in the main DataFrame as well.

df["NFP"] = NFP


# ============================================================
# 5. Construct sequence labels
# ============================================================

labels = []


# ------------------------------------------------------------
# 1-200
# ------------------------------------------------------------

labels += [
    "SELEX"
] * 200


# ------------------------------------------------------------
# 201-400
# ------------------------------------------------------------

labels += [
    "Random"
] * 200


# ------------------------------------------------------------
# 401-600
# ------------------------------------------------------------

labels += [
    "Yeast"
] * 200


# ------------------------------------------------------------
# 601-800
#
# G1-G10, 20 sequences per group
# ------------------------------------------------------------

for group_number in range(
    1,
    11
):

    labels += [
        f"G{group_number}"
    ] * 20


# ------------------------------------------------------------
# 801
# ------------------------------------------------------------

labels += [
    "Widom 601"
]


# ------------------------------------------------------------
# 802-901
# ------------------------------------------------------------

labels += [
    "Yeast"
] * 100


# ------------------------------------------------------------
# 902-1001
# ------------------------------------------------------------

labels += [
    "Mimic yeast (7)"
] * 100


# ------------------------------------------------------------
# 1002-1021
# ------------------------------------------------------------

labels += [
    "Gaussian flex"
] * 20


# ------------------------------------------------------------
# 1022-1041
# ------------------------------------------------------------

labels += [
    "Native 510 mimic"
] * 20


# ------------------------------------------------------------
# 1042-1061
# ------------------------------------------------------------

labels += [
    "Native 545 mimic"
] * 20


# ------------------------------------------------------------
# 1062-1171
# ------------------------------------------------------------

labels += [
    "Codon-randomized 345"
] * 110


# ------------------------------------------------------------
# 1172
# ------------------------------------------------------------

labels += [
    "Native 345"
]


# ------------------------------------------------------------
# 1173-2000
# ------------------------------------------------------------

labels += [
    "Random"
] * 828


assert len(labels) == 2000


# ============================================================
# 6. Construct Figure 3.5 data
# ============================================================

figure_3_5 = pd.DataFrame({

    "Library_position":
        np.arange(
            1,
            2001
        ),

    "Sequence":
        seq,

    "Label":
        labels,

    "A1":
        df["A1"].to_numpy(),

    "A2 (Free)":
        df[
            "A2 (Free)"
        ].to_numpy(),

    "NFP":
        NFP
})


# ============================================================
# 7. Add a broader sequence class
# ============================================================

# This is useful when several related subgroups should be
# considered together in subsequent analyses.


def broad_sequence_class(label):

    if label.startswith("G") and label[1:].isdigit():

        return "G1-G10"

    if label in {
        "Gaussian flex",
        "Native 510 mimic",
        "Native 545 mimic"
    }:

        return "Cyclisability-designed"

    if label in {
        "Codon-randomized 345",
        "Native 345"
    }:

        return "Codon experiment"

    return label


figure_3_5[
    "Broad_class"
] = figure_3_5[
    "Label"
].apply(
    broad_sequence_class
)


# ============================================================
# 8. Group-level summary
# ============================================================

figure_3_5_summary = (

    figure_3_5

    .groupby(
        "Label",
        sort=False
    )

    ["NFP"]

    .agg(
        N_sequences="count",
        Mean_NFP="mean",
        Median_NFP="median",
        SD_NFP="std"
    )

    .reset_index()
)


# ============================================================
# 9. Useful specific subsets
# ============================================================

# 200 G1-G10 sequences.

figure_3_5_G1_G10 = (
    figure_3_5.iloc[
        600:800
    ].copy()
)


assert len(
    figure_3_5_G1_G10
) == 200


# All random sequences:
#
#     200 sequences at positions 201-400
#     +
#     828 sequences at positions 1173-2000
#     =
#     1,028 sequences.

figure_3_5_random = (
    figure_3_5[
        figure_3_5["Label"]
        == "Random"
    ]
    .copy()
)


assert len(
    figure_3_5_random
) == 1028


# All native yeast sequences carrying the general
# "Yeast" label:
#
#     200 at positions 401-600
#     +
#     100 at positions 802-901
#     =
#     300.

figure_3_5_yeast = (
    figure_3_5[
        figure_3_5["Label"]
        == "Yeast"
    ]
    .copy()
)


assert len(
    figure_3_5_yeast
) == 300


# Codon-randomized sequence family and its corresponding
# native sequence.

figure_3_5_codon = (
    figure_3_5[
        figure_3_5["Label"].isin(
            [
                "Codon-randomized 345",
                "Native 345"
            ]
        )
    ]
    .copy()
)


assert len(
    figure_3_5_codon
) == 111


# ============================================================
# 10. Check expected group sizes
# ============================================================

expected_counts = {

    "SELEX": 200,
    "Random": 1028,
    "Yeast": 300,

    "G1": 20,
    "G2": 20,
    "G3": 20,
    "G4": 20,
    "G5": 20,
    "G6": 20,
    "G7": 20,
    "G8": 20,
    "G9": 20,
    "G10": 20,

    "Widom 601": 1,

    "Mimic yeast (7)": 100,

    "Gaussian flex": 20,

    "Native 510 mimic": 20,

    "Native 545 mimic": 20,

    "Codon-randomized 345": 110,

    "Native 345": 1
}


observed_counts = (
    figure_3_5[
        "Label"
    ]
    .value_counts()
    .to_dict()
)


assert observed_counts == expected_counts


# ============================================================
# 11. Final output
# ============================================================

print(
    "Number of sequences:",
    len(
        figure_3_5
    )
)


print(
    "\nNFP calculation:"
)


print(
    "ln[(A1 + 0.01) / "
    "(A2 (Free) + 0.01)]"
)


print(
    "\nSequence groups:"
)


print(
    figure_3_5_summary.to_string(
        index=False
    )
)


# ============================================================
# OUTPUT SUMMARY
# ============================================================

# Ordered sequence list:
#
#     seq
#
#
# Measured NFP:
#
#     NFP
#
#
# Ordered sequence labels:
#
#     labels
#
#
# Complete data underlying Figure 3.5:
#
#     figure_3_5
#
#
# Group summary:
#
#     figure_3_5_summary
#
#
# Useful subsets:
#
#     figure_3_5_G1_G10
#     figure_3_5_random
#     figure_3_5_yeast
#     figure_3_5_codon
#
#
# No plotting is performed.