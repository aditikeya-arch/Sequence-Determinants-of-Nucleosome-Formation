

"""
Code 25 Figure 3-4 G1-G10 intrinsic cyclisability profiles.py
=============================================================

WHAT THIS SCRIPT DOES
---------------------
Generates the intrinsic-cyclisability data underlying
Figure 3.4.

The 2,000-member competitive nucleosome reconstitution
library contains 200 designed sequences belonging to
groups G1-G10.

Each group contains 20 sequences:

    G1     library positions 601-620
    G2     library positions 621-640
    G3     library positions 641-660
    G4     library positions 661-680
    G5     library positions 681-700
    G6     library positions 701-720
    G7     library positions 721-740
    G8     library positions 741-760
    G9     library positions 761-780
    G10    library positions 781-800

For every 147-bp sequence, intrinsic cyclisability is
predicted for all overlapping 50-bp windows.

A 147-bp sequence therefore produces:

    147 - 50 + 1 = 98

intrinsic-cyclisability values.

The mean profile of the 20 sequences belonging to each
group is then calculated.


VARIABLES THAT MUST ALREADY EXIST
---------------------------------
seq

    Ordered list containing all 2,000 sequences in the
    competitive nucleosome reconstitution library.

    The order must be the experimental library order.

    Each sequence must be 147 bp long.


FUNCTIONS THAT MUST ALREADY EXIST
---------------------------------
load_model

pred

These are the intrinsic-cyclisability prediction
functions used throughout the thesis.

They can be obtained from:

    https://github.com/codergirl1106/
    Cyclizability-Prediction-Website/


INPUT FILES
-----------
None, provided that seq is already defined.

Alternatively, seq can be obtained from the Sequence
column of A1_A6_Counts.csv after removing the final
blank row.


OUTPUTS
-------
c0_mat

    2000 x 98 matrix containing the predicted intrinsic-
    cyclisability profile of every sequence in the
    experimental library.


figure_3_4_profiles

    Dictionary containing the 20 x 98 cyclisability
    matrix for each of G1-G10.


figure_3_4_mean_profiles

    DataFrame containing the mean intrinsic-cyclisability
    profile for each group.

    Columns:

        Position
        G1
        G2
        ...
        G10


figure_3_4_individual_profiles

    Long-format DataFrame containing the individual
    profiles of all 200 G1-G10 sequences.


POSITION CONVENTION
-------------------
The first predicted value corresponds to the 50-bp
window spanning sequence positions 1-50.

Following the convention used throughout the thesis,
this value is assigned nominally to position 25.

The 98 predicted values are therefore assigned to:

    25, 26, ..., 122


OTHER SCRIPTS THAT MUST BE RUN FIRST
------------------------------------
None, provided that seq, load_model, and pred are
already defined.


NOTES
-----
The G1-G10 sequences occupy Python indices 600:800
because Python uses zero-based indexing.

Each group contains exactly 20 consecutive sequences.

No CPM or other weighting is used. Every sequence
contributes equally to its group mean.

No plotting is performed.
"""


import numpy as np
import pandas as pd


# ============================================================
# 1. Constants
# ============================================================

SEQUENCE_LENGTH = 147
WINDOW = 50

PROFILE_LENGTH = (
    SEQUENCE_LENGTH - WINDOW + 1
)  # 98

N_LIBRARY_SEQUENCES = 2000

N_GROUPS = 10
SEQUENCES_PER_GROUP = 20

G_START = 600
G_END = 800


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


# Convert to a normal list of uppercase strings.

seq = [
    str(sequence).upper()
    for sequence in seq
]


assert len(seq) == N_LIBRARY_SEQUENCES, (
    f"Expected {N_LIBRARY_SEQUENCES} sequences, "
    f"but found {len(seq)}."
)


assert all(
    len(sequence) == SEQUENCE_LENGTH
    for sequence in seq
), "Every library sequence must be 147 bp long."


assert all(
    set(sequence).issubset({"A", "C", "G", "T"})
    for sequence in seq
), "Sequences must contain only A, C, G, and T."


# ============================================================
# 3. Predict intrinsic cyclisability for all 2,000 sequences
# ============================================================

model = load_model(0)


seqparsed = [

    sequence[i:i + WINDOW]

    for sequence in seq

    for i in range(PROFILE_LENGTH)
]


# 2,000 sequences x 98 windows each
# = 196,000 predictions.

c0_vals = np.asarray(
    pred(
        model,
        seqparsed
    )
).reshape(-1)


assert c0_vals.size == (
    N_LIBRARY_SEQUENCES
    * PROFILE_LENGTH
)


c0_mat = c0_vals.reshape(
    N_LIBRARY_SEQUENCES,
    PROFILE_LENGTH
)


assert c0_mat.shape == (
    2000,
    98
)


# ============================================================
# 4. Extract the 200 G1-G10 sequences
# ============================================================

c0_mat_G = c0_mat[
    G_START:G_END,
    :
]


assert c0_mat_G.shape == (
    200,
    98
)


# ============================================================
# 5. Divide the 200 sequences into G1-G10
# ============================================================

figure_3_4_profiles = {}


for group_number in range(
    1,
    N_GROUPS + 1
):

    start = (
        (group_number - 1)
        * SEQUENCES_PER_GROUP
    )

    end = (
        start
        + SEQUENCES_PER_GROUP
    )

    group_name = (
        f"G{group_number}"
    )

    figure_3_4_profiles[
        group_name
    ] = c0_mat_G[
        start:end,
        :
    ]


# Check every group.

for group_name, profiles in (
    figure_3_4_profiles.items()
):

    assert profiles.shape == (
        20,
        98
    )


# ============================================================
# 6. Calculate the mean profile of each group
# ============================================================

figure_3_4_mean_profiles = pd.DataFrame({

    "Position":
        np.arange(
            25,
            25 + PROFILE_LENGTH
        )
})


for group_number in range(
    1,
    N_GROUPS + 1
):

    group_name = (
        f"G{group_number}"
    )

    figure_3_4_mean_profiles[
        group_name
    ] = np.mean(

        figure_3_4_profiles[
            group_name
        ],

        axis=0
    )


# ============================================================
# 7. Construct long-format individual-profile data
# ============================================================

individual_rows = []


for group_number in range(
    1,
    N_GROUPS + 1
):

    group_name = (
        f"G{group_number}"
    )

    profiles = (
        figure_3_4_profiles[
            group_name
        ]
    )


    for sequence_in_group in range(
        SEQUENCES_PER_GROUP
    ):

        # Zero-based index in the complete 2,000-member
        # library.

        library_index = (

            G_START

            + (
                group_number - 1
            )
            * SEQUENCES_PER_GROUP

            + sequence_in_group
        )


        # Human-readable library position: 1-2000.

        library_position = (
            library_index + 1
        )


        for profile_index in range(
            PROFILE_LENGTH
        ):

            individual_rows.append({

                "Group":
                    group_name,

                "Sequence_in_group":
                    sequence_in_group + 1,

                "Library_position":
                    library_position,

                "Position":
                    profile_index + 25,

                "Intrinsic_cyclisability":
                    profiles[
                        sequence_in_group,
                        profile_index
                    ]
            })


figure_3_4_individual_profiles = (
    pd.DataFrame(
        individual_rows
    )
)


# ============================================================
# 8. Summary table
# ============================================================

figure_3_4_summary = pd.DataFrame({

    "Group": [
        f"G{i}"
        for i in range(
            1,
            11
        )
    ],

    "First_library_position": [
        601 + 20 * i
        for i in range(10)
    ],

    "Last_library_position": [
        620 + 20 * i
        for i in range(10)
    ],

    "N_sequences": [
        20
        for _ in range(10)
    ]
})


# ============================================================
# 9. Final checks
# ============================================================

assert len(
    figure_3_4_mean_profiles
) == 98


assert len(
    figure_3_4_individual_profiles
) == (
    200
    * 98
)


assert (
    figure_3_4_summary[
        "N_sequences"
    ].sum()
    == 200
)


print(
    "c0_mat shape:",
    c0_mat.shape
)


print(
    "G1-G10 profile matrix shape:",
    c0_mat_G.shape
)


print(
    "\nFigure 3.4 groups:"
)


print(
    figure_3_4_summary.to_string(
        index=False
    )
)


# ============================================================
# OUTPUT SUMMARY
# ============================================================

# Intrinsic-cyclisability profiles for the complete
# 2,000-member library:
#
#     c0_mat
#
# shape:
#
#     2000 x 98
#
#
# Individual G1-G10 matrices:
#
#     figure_3_4_profiles["G1"]
#     figure_3_4_profiles["G2"]
#     ...
#     figure_3_4_profiles["G10"]
#
# Each has shape:
#
#     20 x 98
#
#
# Data underlying the mean curves in Figure 3.4:
#
#     figure_3_4_mean_profiles
#
#
# Individual sequence profiles:
#
#     figure_3_4_individual_profiles
#
#
# Group definitions:
#
#     figure_3_4_summary
#
#
# No plotting is performed.