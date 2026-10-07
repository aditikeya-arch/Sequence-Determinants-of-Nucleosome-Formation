

"""
Code 19 Figure 2-24 HRI contribution to intrinsic cyclisability.py
==================================================================

WHAT THIS SCRIPT DOES
---------------------
Generates the data underlying Figure 2.24.

The analysis asks how the helical spacing of each possible
NN-NN dinucleotide pair relates to intrinsic DNA
cyclisability.

The procedure is:

    1. Generate 1,000,000 random 50-bp DNA sequences.

    2. Predict intrinsic cyclisability for every sequence.

    3. For every individual sequence, calculate the HRI
       value for all 136 unordered NN-NN dinucleotide pairs.

    4. For each NN-NN pair independently, fit:

           intrinsic cyclisability = slope * HRI + intercept

       across the 1,000,000 random sequences.

    5. Store the slope for that NN-NN pair.

    6. Arrange the 136 slopes into a symmetric 16 x 16
       matrix for Figure 2.24.


INTERPRETATION
--------------
Positive slope:

    Increasing HRI for that NN-NN pair is associated with
    increasing intrinsic cyclisability.

Negative slope:

    Increasing HRI for that NN-NN pair is associated with
    decreasing intrinsic cyclisability.

For WW-WW and SS-SS pairs, a positive slope means that
spacing those motifs preferentially at full-helical-repeat
distances tends to promote cyclisability.

For WW-SS pairs, a negative slope means that preference
for half-helical-repeat spacing tends to promote
cyclisability.


HRI DEFINITION
--------------
For a dinucleotide pair, the normalized pairwise spacing
distribution is rho(d).

For each nominal full-helical or half-helical separation,
the maximum rho value within +/- 1 bp of the nominal
separation is used.

The HRI is:

    HRI =
        sum(max rho around full-helical spacings)
        -
        sum(max rho around half-helical spacings)

The same definition used for Figures 2.22 and 2.23 is used
here, except that only separations possible within a
50-bp sequence can contribute.


IMPORTANT DIFFERENCE FROM FIGURES 2.22-2.23
-------------------------------------------
For Figures 2.22-2.23, pair counts were pooled across a
group of sequences before calculating HRI.

For Figure 2.24, HRI is calculated separately for every
individual sequence.

This is necessary because the analysis regresses intrinsic
cyclisability against HRI across sequences.


INPUT FILES
-----------
None.


VARIABLES / FUNCTIONS THAT MUST ALREADY EXIST
---------------------------------------------
load_model
pred

These are the external intrinsic-cyclisability prediction
functions used throughout the analysis.


OTHER SCRIPTS THAT MUST BE RUN FIRST
------------------------------------
None.

Code 17 Figure 2-22 HRI heatmaps.py contains the related
group-level HRI calculation, but Figure 2.24 requires an
individual-sequence implementation because the sequence
length is 50 bp.


OUTPUTS
-------
figure_2_24

    16 x 16 symmetric DataFrame containing the regression
    slope for every NN-NN pair.


figure_2_24_long

    Long-format version containing:

        Dinucleotide_1
        Dinucleotide_2
        Slope


figure_2_24_regression

    One row for each of the 136 independent NN-NN pairs,
    containing:

        Dinucleotide_1
        Dinucleotide_2
        Slope
        Intercept
        Pearson_r


No plotting is performed.
"""


import numpy as np
import pandas as pd


# ============================================================
# 1. Analysis settings
# ============================================================

N_RANDOM_SEQUENCES = 1_000_000

SEQUENCE_LENGTH = 50


BASES = np.array(
    [
        "A",
        "C",
        "G",
        "T",
    ]
)


# Fixed seed for reproducibility.

RANDOM_SEED = 1


# Process sequences in batches so that one million
# sequences do not need to be represented by a single
# very large intermediate object.

BATCH_SIZE = 50_000


# ============================================================
# 2. HRI settings
# ============================================================

# A 50-bp sequence contains 49 dinucleotide start
# positions.

N_DINUCLEOTIDE_POSITIONS = (
    SEQUENCE_LENGTH - 1
)


# Full-helical-repeat centres.

HELIX_CENTRES = np.arange(
    10,
    50,
    10
)


# Odd multiples of approximately half a helical repeat.

HALF_HELIX_CENTRES = np.arange(
    5,
    50,
    10
)


# ============================================================
# 3. Dinucleotides
# ============================================================

DINUCS = [

    a + b

    for a in "ACGT"

    for b in "ACGT"
]


# 136 independent unordered pairs.

PAIRS = [

    (i, j)

    for i in range(16)

    for j in range(
        i,
        16
    )
]


assert len(
    PAIRS
) == 136


# ============================================================
# 4. Figure ordering
# ============================================================

DINUCLEOTIDE_ORDER = [

    "AA",
    "AT",
    "TA",
    "TT",

    "AC",
    "AG",
    "CA",
    "CT",
    "GA",
    "GT",
    "TC",
    "TG",

    "CC",
    "CG",
    "GC",
    "GG",
]


ORDER_INDEX = [

    DINUCS.index(
        dinucleotide
    )

    for dinucleotide
    in DINUCLEOTIDE_ORDER
]


# ============================================================
# 5. Encode bases and dinucleotides
# ============================================================

BASE_TO_INT = {

    "A": 0,

    "C": 1,

    "G": 2,

    "T": 3,
}


# ============================================================
# 6. Generate random 50-bp sequences
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)


random_base_codes = rng.integers(

    0,
    4,

    size=(
        N_RANDOM_SEQUENCES,
        SEQUENCE_LENGTH
    ),

    dtype=np.uint8
)


# Convert to strings for the cyclisability model.

base_lookup = np.array(
    [
        "A",
        "C",
        "G",
        "T",
    ]
)


random_sequences = [

    "".join(
        base_lookup[
            row
        ]
    )

    for row in random_base_codes
]


# ============================================================
# 7. Predict intrinsic cyclisability
# ============================================================

print(
    "Loading intrinsic cyclisability model..."
)


model = load_model(
    0
)


print(
    "Predicting intrinsic cyclisability..."
)


cyclisability = np.empty(
    N_RANDOM_SEQUENCES,
    dtype=np.float64
)


for start in range(
    0,
    N_RANDOM_SEQUENCES,
    BATCH_SIZE
):

    stop = min(

        start + BATCH_SIZE,

        N_RANDOM_SEQUENCES
    )


    print(
        "Cyclisability:",
        start,
        "to",
        stop
    )


    cyclisability[
        start:
        stop
    ] = np.asarray(

        pred(
            model,
            random_sequences[
                start:
                stop
            ]
        )

    ).reshape(-1)


# ============================================================
# 8. Convert base arrays to dinucleotide codes
# ============================================================

# Code:
#
#     4 * first_base + second_base
#
# gives values 0-15 corresponding to DINUCS.

dinucleotide_codes = (

    4
    *
    random_base_codes[
        :,
        :-1
    ]

    +

    random_base_codes[
        :,
        1:
    ]
)


# Shape:
#
#     1,000,000 x 49


# ============================================================
# 9. Valid HRI windows
# ============================================================

def valid_window(
    centre
):

    """
    Return the valid separations within centre +/- 1
    that can occur in a 50-bp sequence.
    """

    return [

        distance

        for distance in (
            centre - 1,
            centre,
            centre + 1
        )

        if (
            distance >= 1

            and

            distance
            <
            N_DINUCLEOTIDE_POSITIONS
        )
    ]


FULL_WINDOWS = [

    valid_window(
        centre
    )

    for centre in HELIX_CENTRES
]


HALF_WINDOWS = [

    valid_window(
        centre
    )

    for centre in HALF_HELIX_CENTRES
]


# Remove any empty windows.

FULL_WINDOWS = [

    window

    for window in FULL_WINDOWS

    if len(
        window
    ) > 0
]


HALF_WINDOWS = [

    window

    for window in HALF_WINDOWS

    if len(
        window
    ) > 0
]


print(
    "Full-helical HRI windows:",
    FULL_WINDOWS
)


print(
    "Half-helical HRI windows:",
    HALF_WINDOWS
)


# ============================================================
# 10. Expected random probability
# ============================================================

def expected_pair_probability(
    dinucleotide_1,
    dinucleotide_2
):

    """
    Expected probability under random equal-base DNA.

    Identical unordered pair:

        1 / 256

    Different unordered pair:

        2 / 256
        =
        1 / 128
    """

    if (
        dinucleotide_1
        ==
        dinucleotide_2
    ):

        return 1.0 / 256.0


    return 2.0 / 256.0


# ============================================================
# 11. Calculate rho for one pair and one distance
# ============================================================

def pair_rho(
    dinucleotide_matrix,
    dinucleotide_1,
    dinucleotide_2,
    distance
):

    """
    Calculate rho(d) independently for every sequence.

    Returns one value per sequence.
    """

    left = (

        dinucleotide_matrix[
            :,
            :-distance
        ]
    )


    right = (

        dinucleotide_matrix[
            :,
            distance:
        ]
    )


    if (
        dinucleotide_1
        ==
        dinucleotide_2
    ):

        observed = np.sum(

            (
                left
                ==
                dinucleotide_1
            )

            &

            (
                right
                ==
                dinucleotide_2
            ),

            axis=1
        )


    else:

        forward = (

            (
                left
                ==
                dinucleotide_1
            )

            &

            (
                right
                ==
                dinucleotide_2
            )
        )


        reverse = (

            (
                left
                ==
                dinucleotide_2
            )

            &

            (
                right
                ==
                dinucleotide_1
            )
        )


        observed = np.sum(

            forward
            |
            reverse,

            axis=1
        )


    possible_position_pairs = (

        N_DINUCLEOTIDE_POSITIONS

        -

        distance
    )


    expected = (

        possible_position_pairs

        *

        expected_pair_probability(
            dinucleotide_1,
            dinucleotide_2
        )
    )


    return (

        observed.astype(
            np.float64
        )

        /

        expected
    )


# ============================================================
# 12. Calculate individual-sequence HRI
# ============================================================

def calculate_pair_hri(
    dinucleotide_matrix,
    dinucleotide_1,
    dinucleotide_2
):

    """
    Calculate the HRI of one unordered NN-NN pair
    independently for every sequence.
    """

    n_sequences = (
        dinucleotide_matrix.shape[0]
    )


    full_sum = np.zeros(
        n_sequences,
        dtype=np.float64
    )


    half_sum = np.zeros(
        n_sequences,
        dtype=np.float64
    )


    # --------------------------------------------------------
    # Full-helical-repeat contribution
    # --------------------------------------------------------

    for window in FULL_WINDOWS:

        rho_values = np.column_stack(

            [

                pair_rho(
                    dinucleotide_matrix,
                    dinucleotide_1,
                    dinucleotide_2,
                    distance
                )

                for distance in window
            ]
        )


        full_sum += np.max(
            rho_values,
            axis=1
        )


    # --------------------------------------------------------
    # Half-helical-repeat contribution
    # --------------------------------------------------------

    for window in HALF_WINDOWS:

        rho_values = np.column_stack(

            [

                pair_rho(
                    dinucleotide_matrix,
                    dinucleotide_1,
                    dinucleotide_2,
                    distance
                )

                for distance in window
            ]
        )


        half_sum += np.max(
            rho_values,
            axis=1
        )


    return (
        full_sum
        -
        half_sum
    )


# ============================================================
# 13. Linear regression helper
# ============================================================

def regression_statistics(
    x,
    y
):

    """
    Ordinary least-squares regression:

        y = slope*x + intercept

    Also returns Pearson's r.
    """

    x = np.asarray(
        x,
        dtype=np.float64
    )


    y = np.asarray(
        y,
        dtype=np.float64
    )


    x_mean = np.mean(
        x
    )


    y_mean = np.mean(
        y
    )


    dx = (
        x
        -
        x_mean
    )


    dy = (
        y
        -
        y_mean
    )


    denominator = np.sum(
        dx ** 2
    )


    if denominator == 0:

        return (
            np.nan,
            np.nan,
            np.nan
        )


    slope = (

        np.sum(
            dx
            *
            dy
        )

        /

        denominator
    )


    intercept = (

        y_mean

        -

        slope
        *
        x_mean
    )


    correlation_denominator = np.sqrt(

        np.sum(
            dx ** 2
        )

        *

        np.sum(
            dy ** 2
        )
    )


    if correlation_denominator == 0:

        pearson_r = np.nan


    else:

        pearson_r = (

            np.sum(
                dx
                *
                dy
            )

            /

            correlation_denominator
        )


    return (
        slope,
        intercept,
        pearson_r
    )


# ============================================================
# 14. Calculate slopes for all 136 NN-NN pairs
# ============================================================

regression_rows = []


slope_matrix = np.zeros(
    (
        16,
        16
    ),
    dtype=np.float64
)


for pair_number, (
    dinucleotide_1,
    dinucleotide_2
) in enumerate(
    PAIRS,
    start=1
):

    print(
        "Pair",
        pair_number,
        "of",
        len(PAIRS),
        ":",
        DINUCS[
            dinucleotide_1
        ],
        "-",
        DINUCS[
            dinucleotide_2
        ]
    )


    # --------------------------------------------------------
    # Individual-sequence HRI
    # --------------------------------------------------------

    hri = calculate_pair_hri(

        dinucleotide_codes,

        dinucleotide_1,

        dinucleotide_2
    )


    # --------------------------------------------------------
    # Regression against intrinsic cyclisability
    # --------------------------------------------------------

    (
        slope,
        intercept,
        pearson_r

    ) = regression_statistics(

        hri,

        cyclisability
    )


    # --------------------------------------------------------
    # Store symmetric matrix
    # --------------------------------------------------------

    slope_matrix[
        dinucleotide_1,
        dinucleotide_2
    ] = slope


    slope_matrix[
        dinucleotide_2,
        dinucleotide_1
    ] = slope


    regression_rows.append({

        "Dinucleotide_1":
            DINUCS[
                dinucleotide_1
            ],

        "Dinucleotide_2":
            DINUCS[
                dinucleotide_2
            ],

        "Slope":
            slope,

        "Intercept":
            intercept,

        "Pearson_r":
            pearson_r,
    })


figure_2_24_regression = pd.DataFrame(
    regression_rows
)


# ============================================================
# 15. Reorder slope matrix for Figure 2.24
# ============================================================

slope_matrix_ordered = (

    slope_matrix[
        np.ix_(
            ORDER_INDEX,
            ORDER_INDEX
        )
    ]
)


figure_2_24 = pd.DataFrame(

    slope_matrix_ordered,

    index=DINUCLEOTIDE_ORDER,

    columns=DINUCLEOTIDE_ORDER
)


# ============================================================
# 16. Long-format output
# ============================================================

figure_2_24_long_rows = []


for dinucleotide_1 in DINUCLEOTIDE_ORDER:

    for dinucleotide_2 in DINUCLEOTIDE_ORDER:

        figure_2_24_long_rows.append({

            "Dinucleotide_1":
                dinucleotide_1,

            "Dinucleotide_2":
                dinucleotide_2,

            "Slope":
                figure_2_24.loc[
                    dinucleotide_1,
                    dinucleotide_2
                ],
        })


figure_2_24_long = pd.DataFrame(
    figure_2_24_long_rows
)


# ============================================================
# OUTPUT SUMMARY
# ============================================================

# Figure 2.24 heatmap matrix:
#
#     figure_2_24
#
# 16 x 16 symmetric matrix containing the slope:
#
#     intrinsic cyclisability
#              vs
#     individual-sequence HRI
#
#
# Independent-pair regression table:
#
#     figure_2_24_regression
#
# 136 rows containing:
#
#     Dinucleotide_1
#     Dinucleotide_2
#     Slope
#     Intercept
#     Pearson_r
#
#
# Long-format heatmap table:
#
#     figure_2_24_long
#
#
# Expected qualitative pattern:
#
#     WW-WW:
#         predominantly positive slopes
#
#     SS-SS:
#         predominantly positive slopes
#
#     WW-SS:
#         predominantly negative slopes
#
#
# No plotting is performed.