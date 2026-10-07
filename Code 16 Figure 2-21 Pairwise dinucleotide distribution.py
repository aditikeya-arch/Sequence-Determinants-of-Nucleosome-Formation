

"""
Code 16 Figure 2-21 Pairwise dinucleotide distribution.py
==========================================================

WHAT THIS SCRIPT DOES
---------------------
Generates the data underlying Figure 2.21.

Figure 2.21 shows pairwise dinucleotide spacing distributions
for representative dinucleotide pairs among the most enriched
Round 6 SELEX sequences.

The pairs shown are:

    TT-TT
    GG-GG
    TT-GG


For each 147-bp SELEX sequence:

    1. The central 113-bp variable region is retained.

    2. All adjacent dinucleotides are identified.

       A 113-bp sequence contains 112 possible
       dinucleotide start positions.

    3. For each separation i, occurrences of the selected
       dinucleotide pair are counted.

       Separation is defined as the difference between
       dinucleotide start positions.

       For example:

           separation = 10

       means that the two dinucleotide start positions
       differ by 10 bp.

    4. Pair counts are pooled across the complete group
       of sequences.

    5. For a given separation i, the number of possible
       start-position pairs in one 113-bp sequence is:

           112 - i

    6. The pooled pair count is divided by the total
       number of possible position pairs.

    7. This value is further normalized by the expected
       probability under a random DNA model with equal
       frequencies of A, C, G and T.

       For identical dinucleotide pairs:

           expected probability = 1 / 256

       For different unordered dinucleotide pairs:

           expected probability = 2 / 256
                                = 1 / 128

       For a non-identical pair such as TT-GG, both
       orientations are therefore counted:

           TT ... GG

       and:

           GG ... TT


INTERPRETATION
--------------
A normalized pairwise distribution value of:

    1

means that the dinucleotide pair occurs at that separation
at the frequency expected for random DNA with equal base
frequencies.

Values above 1 indicate enrichment.

Values below 1 indicate depletion.


FIGURE 2.21
-----------
The analysis is performed for:

    TT-TT
    GG-GG
    TT-GG

among the most enriched Round 6 SELEX sequences.


INPUT FILE
----------
R6.csv

The CSV must contain:

    Sequence
    Counts Per Million


TOP-SEQUENCE FILTERING
----------------------
Sequences are ranked by Counts Per Million.

The top 10,000 sequences are initially selected.

Near-duplicate sequences differing at fewer than 10 positions
are removed, retaining the higher-abundance sequence.

For the Round 6 dataset this gives 9,803 retained sequences.


OUTPUT
------
figure_2_21

Columns:

    Pair
    Dinucleotide_1
    Dinucleotide_2
    Separation_bp
    Pair_count
    Possible_position_pairs
    Expected_random_probability
    Expected_random_count
    Pairwise_distribution


No plotting is performed.
"""


import numpy as np
import pandas as pd


# ============================================================
# 1. Analysis settings
# ============================================================

INPUT_FILE = "R6.csv"


TOP_N = 10_000

MIN_HAMMING_DISTANCE = 10


RAW_LENGTH = 147

TRIM = 17

VARIABLE_LENGTH = 113


# A 113-bp sequence contains 112 dinucleotide starts.

N_DINUCLEOTIDE_POSITIONS = (
    VARIABLE_LENGTH - 1
)


PAIRS = [

    ("TT", "TT"),

    ("GG", "GG"),

    ("TT", "GG"),
]


# ============================================================
# 2. Load Round 6
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


required_columns = {
    "Sequence",
    "Counts Per Million",
}


missing_columns = (

    required_columns

    -

    set(
        df.columns
    )
)


if missing_columns:

    raise ValueError(
        f"{INPUT_FILE} is missing columns: "
        f"{missing_columns}"
    )


df["Sequence"] = (

    df["Sequence"]

    .astype(str)

    .str.upper()
)


# ============================================================
# 3. Check sequence lengths
# ============================================================

sequence_lengths = (
    df["Sequence"]
    .str.len()
)


if not (
    sequence_lengths == RAW_LENGTH
).all():

    bad_lengths = (

        sequence_lengths[
            sequence_lengths != RAW_LENGTH
        ]

        .value_counts()

        .to_dict()
    )


    raise ValueError(
        "All sequences must be 147 bp long. "
        f"Unexpected lengths found: {bad_lengths}"
    )


# ============================================================
# 4. Hamming distance
# ============================================================

def hamming_distance(
    sequence_1,
    sequence_2
):

    """
    Number of positions at which two equal-length
    sequences differ.
    """

    return sum(

        base_1 != base_2

        for base_1, base_2
        in zip(
            sequence_1,
            sequence_2
        )
    )


# ============================================================
# 5. Select and filter top Round 6 sequences
# ============================================================

def filter_top_sequences(
    dataframe,
    top_n=TOP_N,
    min_distance=MIN_HAMMING_DISTANCE
):

    """
    Rank sequences by Counts Per Million and initially
    select the top N.

    Sequences are processed from highest to lowest
    abundance.

    A sequence is retained only if it differs from every
    previously retained sequence at at least
    `min_distance` positions.
    """

    top = (

        dataframe

        .sort_values(
            "Counts Per Million",
            ascending=False
        )

        .head(
            top_n
        )

        .reset_index(
            drop=True
        )
    )


    retained_rows = []

    retained_sequences = []


    for _, row in top.iterrows():

        sequence = row[
            "Sequence"
        ]


        keep = True


        for previous_sequence in retained_sequences:

            if hamming_distance(
                sequence,
                previous_sequence
            ) < min_distance:

                keep = False

                break


        if keep:

            retained_rows.append(
                row
            )

            retained_sequences.append(
                sequence
            )


    return (

        pd.DataFrame(
            retained_rows
        )

        .reset_index(
            drop=True
        )
    )


df_top_unique = filter_top_sequences(
    df
)


print(
    "Number of Round 6 sequences retained:",
    len(df_top_unique)
)


if len(
    df_top_unique
) != 9803:

    print(
        "\nWARNING:"
        "\nThe Round 6 filtering did not produce exactly "
        "9,803 sequences."
        "\nCheck that R6.csv corresponds to the dataset "
        "used for this analysis."
    )


# ============================================================
# 6. Extract central 113-bp variable regions
# ============================================================

sequences = (

    df_top_unique[
        "Sequence"
    ]

    .str.slice(
        TRIM,
        RAW_LENGTH - TRIM
    )

    .tolist()
)


if not all(
    len(sequence) == VARIABLE_LENGTH
    for sequence in sequences
):

    raise ValueError(
        "Central sequence extraction did not produce "
        "113-bp sequences."
    )


N_SEQUENCES = len(
    sequences
)


# ============================================================
# 7. Convert sequences to dinucleotide arrays
# ============================================================

def sequence_dinucleotides(
    sequence
):

    """
    Return the 112 adjacent dinucleotides from a
    113-bp sequence.

    Element i corresponds to the dinucleotide starting
    at position i.
    """

    return np.asarray(

        [

            sequence[
                i:
                i + 2
            ]

            for i in range(
                N_DINUCLEOTIDE_POSITIONS
            )
        ],

        dtype="<U2"
    )


dinucleotide_arrays = [

    sequence_dinucleotides(
        sequence
    )

    for sequence in sequences
]


# ============================================================
# 8. Expected random probability
# ============================================================

def expected_pair_probability(
    dinucleotide_1,
    dinucleotide_2
):

    """
    Expected probability of an unordered dinucleotide
    pair under random DNA with equal base frequencies.

    Each individual dinucleotide has probability 1/16.

    Identical pair:

        (1/16) * (1/16)
        = 1/256

    Different unordered pair:

        2 * (1/16) * (1/16)
        = 2/256
        = 1/128
    """

    if (
        dinucleotide_1
        ==
        dinucleotide_2
    ):

        return 1.0 / 256.0


    return 2.0 / 256.0


# ============================================================
# 9. Count one pair at one separation
# ============================================================

def count_pair_at_separation(
    dinucleotides,
    dinucleotide_1,
    dinucleotide_2,
    separation
):

    """
    Count occurrences of a dinucleotide pair at a
    specified separation within one sequence.

    For identical pairs, only that orientation exists.

    For different pairs, both orientations are counted:

        dinucleotide_1 ... dinucleotide_2

    and:

        dinucleotide_2 ... dinucleotide_1
    """

    left = (
        dinucleotides[
            :-separation
        ]
    )


    right = (
        dinucleotides[
            separation:
        ]
    )


    if (
        dinucleotide_1
        ==
        dinucleotide_2
    ):

        matches = (

            (left == dinucleotide_1)

            &

            (right == dinucleotide_2)
        )


    else:

        forward = (

            (left == dinucleotide_1)

            &

            (right == dinucleotide_2)
        )


        reverse = (

            (left == dinucleotide_2)

            &

            (right == dinucleotide_1)
        )


        matches = (
            forward
            |
            reverse
        )


    return int(
        np.sum(
            matches
        )
    )


# ============================================================
# 10. Pairwise distribution for one dinucleotide pair
# ============================================================

def pairwise_distribution(
    dinucleotide_1,
    dinucleotide_2
):

    """
    Calculate the pooled normalized pairwise spacing
    distribution for one unordered dinucleotide pair.
    """

    expected_probability = (
        expected_pair_probability(
            dinucleotide_1,
            dinucleotide_2
        )
    )


    rows = []


    # Separation 0 is not considered because it refers to
    # the same dinucleotide start position.

    for separation in range(
        1,
        N_DINUCLEOTIDE_POSITIONS
    ):

        pooled_pair_count = 0


        # ----------------------------------------------------
        # Pool pair counts over all sequences
        # ----------------------------------------------------

        for dinucleotides in dinucleotide_arrays:

            pooled_pair_count += (
                count_pair_at_separation(
                    dinucleotides,
                    dinucleotide_1,
                    dinucleotide_2,
                    separation
                )
            )


        # ----------------------------------------------------
        # Number of possible start-position pairs
        # ----------------------------------------------------

        possible_per_sequence = (

            N_DINUCLEOTIDE_POSITIONS

            -

            separation
        )


        possible_position_pairs = (

            N_SEQUENCES

            *

            possible_per_sequence
        )


        # ----------------------------------------------------
        # Expected count under equal-base random DNA
        # ----------------------------------------------------

        expected_random_count = (

            possible_position_pairs

            *

            expected_probability
        )


        # ----------------------------------------------------
        # Normalized pairwise distribution
        # ----------------------------------------------------

        normalized_distribution = (

            pooled_pair_count

            /

            expected_random_count
        )


        rows.append({

            "Pair":
                (
                    f"{dinucleotide_1}-"
                    f"{dinucleotide_2}"
                ),

            "Dinucleotide_1":
                dinucleotide_1,

            "Dinucleotide_2":
                dinucleotide_2,

            "Separation_bp":
                separation,

            "Pair_count":
                pooled_pair_count,

            "Possible_pairs_per_sequence":
                possible_per_sequence,

            "Possible_position_pairs":
                possible_position_pairs,

            "Expected_random_probability":
                expected_probability,

            "Expected_random_count":
                expected_random_count,

            "Pairwise_distribution":
                normalized_distribution,
        })


    return pd.DataFrame(
        rows
    )


# ============================================================
# 11. Calculate Figure 2.21 pairwise distributions
# ============================================================

pairwise_tables = []


for (
    dinucleotide_1,
    dinucleotide_2
) in PAIRS:

    print(
        "Calculating",
        f"{dinucleotide_1}-{dinucleotide_2}"
    )


    pair_table = pairwise_distribution(
        dinucleotide_1,
        dinucleotide_2
    )


    pairwise_tables.append(
        pair_table
    )


figure_2_21 = pd.concat(
    pairwise_tables,
    ignore_index=True
)


# ============================================================
# 12. Separate outputs for each curve
# ============================================================

figure_2_21_TT_TT = (

    figure_2_21.loc[
        figure_2_21[
            "Pair"
        ] == "TT-TT"
    ]

    .reset_index(
        drop=True
    )
)


figure_2_21_GG_GG = (

    figure_2_21.loc[
        figure_2_21[
            "Pair"
        ] == "GG-GG"
    ]

    .reset_index(
        drop=True
    )
)


figure_2_21_TT_GG = (

    figure_2_21.loc[
        figure_2_21[
            "Pair"
        ] == "TT-GG"
    ]

    .reset_index(
        drop=True
    )
)


# ============================================================
# 13. Wide-format table
# ============================================================

figure_2_21_wide = (

    figure_2_21

    .pivot(
        index="Separation_bp",
        columns="Pair",
        values="Pairwise_distribution"
    )

    .reset_index()
)


# ============================================================
# OUTPUT SUMMARY
# ============================================================

# figure_2_21
#
# contains the normalized pairwise spacing distributions
# for:
#
#     TT-TT
#     GG-GG
#     TT-GG
#
#
# Separate dataframes:
#
#     figure_2_21_TT_TT
#     figure_2_21_GG_GG
#     figure_2_21_TT_GG
#
#
# Wide-format dataframe:
#
#     figure_2_21_wide
#
#
# Pairwise_distribution = 1:
#
#     occurrence at the frequency expected under
#     equal-base random DNA.
#
#
# Pairwise_distribution > 1:
#
#     enrichment at that separation.
#
#
# Pairwise_distribution < 1:
#
#     depletion at that separation.
#
#
# No plotting is performed.