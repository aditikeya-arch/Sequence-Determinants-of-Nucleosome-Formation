

"""
SCRIPT NAME:
Code 37 Figure 3-16b NCP score and predicted occupancy.py


WHAT THIS SCRIPT DOES:

Reconstructs Figure 3.16b.

For every experimentally mapped native yeast nucleosome:

    1. Takes its published in-vivo Nucleosome Centre
       Positioning (NCP) score.

    2. Uses its SacCer3 dyad coordinate obtained by liftOver.

    3. Reads the predicted nucleosome occupancy at that genomic
       position from the 10-mer cumulative-score + hard-rod model.

    4. Compares NCP score with predicted occupancy.

The script generates the data for:

    LEFT PANEL:
        individual native nucleosomes:
        NCP score vs predicted occupancy

    RIGHT PANEL:
        binned NCP score vs mean predicted occupancy,
        with SEM.

No plotting is performed.


THESIS SECTION / FIGURE:

Figure 3.16b


INPUT / SOURCE OF NCP SCORE:

The NCP score comes from:

    NucleosomeSacCer2.txt

and is carried through the SacCer2 -> SacCer3 liftOver in:

    Code 5 Native yeast nucleosomal sequences.py


Code 5 produces:

    native_nucleosomes

with columns:

    Chromosome
    Dyad
    NCP_score
    Sequence


IMPORTANT:

The NCP score is NOT calculated here.

It is the experimentally derived/published NCP score associated
with the mapped native yeast nucleosome.

Code 5 preserves its association with the correct nucleosome
during liftOver by using a unique Nucleosome_ID.


VARIABLES THAT MUST ALREADY EXIST:

native_nucleosomes

    Produced by:

        Code 5 Native yeast nucleosomal sequences.py

    Required columns:

        Chromosome
        Dyad
        NCP_score


yeast_pred_occ

    Produced by:

        Code 34 Figure 3-15 genome-wide nucleosome occupancy.py

    Dictionary:

        yeast_pred_occ["chr1"]
        ...
        yeast_pred_occ["chr16"]

    containing the predicted per-base nucleosome occupancy from
    the 108-bp 10-mer cumulative-score + hard-rod model.


OTHER SCRIPTS THAT MUST BE RUN FIRST:

Code 5 Native yeast nucleosomal sequences.py

Code 34 Figure 3-15 genome-wide nucleosome occupancy.py


OUTPUT VARIABLES:

figure_3_16b_scatter

    One row per native nucleosome containing:

        chromosome
        SacCer3 dyad
        NCP score
        predicted occupancy


figure_3_16b_binned

    Binned NCP-score data containing:

        mean NCP score
        mean predicted occupancy
        SEM predicted occupancy
        number of nucleosomes


figure_3_16b_pearson_r

    Pearson correlation coefficient between NCP score and
    predicted occupancy.


NOTES:

The SacCer3 dyad coordinates in native_nucleosomes originate
from the BED/liftOver procedure in Code 5.

They are therefore used directly as Python genomic-array indices.

No "-1" conversion is applied here.
"""


import numpy as np
import pandas as pd

from scipy.stats import pearsonr


# ================================================================
# 1. Check required variables
# ================================================================

if "native_nucleosomes" not in globals():

    raise NameError(
        "native_nucleosomes is not defined. "
        "Run Code 5 Native yeast nucleosomal sequences.py first."
    )


if "yeast_pred_occ" not in globals():

    raise NameError(
        "yeast_pred_occ is not defined. "
        "Run Code 34 Figure 3-15 genome-wide "
        "nucleosome occupancy.py first."
    )


required_columns = {
    "Chromosome",
    "Dyad",
    "NCP_score"
}


missing_columns = (
    required_columns
    -
    set(native_nucleosomes.columns)
)


if missing_columns:

    raise ValueError(
        "native_nucleosomes is missing columns: "
        +
        ", ".join(
            sorted(missing_columns)
        )
    )


# ================================================================
# 2. Convert chromosome names
# ================================================================
#
# Code 5 retains UCSC SacCer3 chromosome names:
#
#     chrI
#     chrII
#     ...
#     chrXVI
#
# whereas yeast_pred_occ from Code 34 uses:
#
#     chr1
#     chr2
#     ...
#     chr16
#
# ================================================================

ROMAN_TO_ARABIC = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
    "XIII": 13,
    "XIV": 14,
    "XV": 15,
    "XVI": 16
}


def chromosome_to_predicted_key(chromosome):

    chromosome = str(
        chromosome
    ).strip()


    if chromosome.lower().startswith(
        "chr"
    ):

        chromosome_part = (
            chromosome[3:]
        )

    else:

        chromosome_part = (
            chromosome
        )


    # Already numeric.

    if chromosome_part.isdigit():

        chromosome_number = int(
            chromosome_part
        )

        return (
            f"chr{chromosome_number}"
        )


    chromosome_part = (
        chromosome_part.upper()
    )


    if (
        chromosome_part
        not in
        ROMAN_TO_ARABIC
    ):

        raise ValueError(
            "Unrecognised chromosome identifier: "
            +
            str(chromosome)
        )


    chromosome_number = (
        ROMAN_TO_ARABIC[
            chromosome_part
        ]
    )


    return (
        f"chr{chromosome_number}"
    )


# ================================================================
# 3. Prepare native nucleosome table
# ================================================================

df_native = (
    native_nucleosomes[
        [
            "Chromosome",
            "Dyad",
            "NCP_score"
        ]
    ]
    .copy()
)


df_native["Predicted_chromosome"] = (
    df_native["Chromosome"]
    .apply(
        chromosome_to_predicted_key
    )
)


# Ensure coordinates and NCP scores are numeric.

df_native["Dyad"] = pd.to_numeric(
    df_native["Dyad"],
    errors="coerce"
)


df_native["NCP_score"] = pd.to_numeric(
    df_native["NCP_score"],
    errors="coerce"
)


df_native = df_native[
    df_native["Dyad"].notna()
    &
    df_native["NCP_score"].notna()
].copy()


df_native["Dyad"] = (
    df_native["Dyad"]
    .astype(int)
)


print(
    "Native nucleosomes with valid dyad and NCP score:",
    len(df_native)
)


# ================================================================
# 4. Read predicted occupancy at each native nucleosome dyad
# ================================================================
#
# IMPORTANT:
#
# Code 5 obtains the SacCer3 dyad from the lifted BED interval:
#
#     Dyad = Start + 10
#
# These coordinates are already in the genomic coordinate system
# used for Python chromosome arrays.
#
# Therefore:
#
#     predicted_occupancy = yeast_pred_occ[chrom][dyad]
#
# and NOT:
#
#     yeast_pred_occ[chrom][dyad - 1]
#
# ================================================================

predicted_occupancies = []


for chrom, dyad in zip(

    df_native[
        "Predicted_chromosome"
    ],

    df_native[
        "Dyad"
    ]
):

    # ------------------------------------------------------------
    # Check chromosome
    # ------------------------------------------------------------

    if chrom not in yeast_pred_occ:

        predicted_occupancies.append(
            np.nan
        )

        continue


    occupancy_array = np.asarray(
        yeast_pred_occ[
            chrom
        ],
        dtype=float
    )


    # ------------------------------------------------------------
    # Check coordinate
    # ------------------------------------------------------------

    if (
        dyad < 0
        or
        dyad >= len(
            occupancy_array
        )
    ):

        predicted_occupancies.append(
            np.nan
        )

        continue


    # ------------------------------------------------------------
    # Predicted occupancy at the native dyad
    # ------------------------------------------------------------

    predicted_occupancies.append(
        occupancy_array[
            dyad
        ]
    )


df_native[
    "Predicted_occupancy"
] = np.asarray(
    predicted_occupancies,
    dtype=float
)


# ================================================================
# 5. Keep nucleosomes with finite values
# ================================================================

valid_mask = (

    np.isfinite(
        df_native[
            "NCP_score"
        ]
    )

    &

    np.isfinite(
        df_native[
            "Predicted_occupancy"
        ]
    )
)


figure_3_16b_scatter = (
    df_native.loc[
        valid_mask,
        [
            "Chromosome",
            "Predicted_chromosome",
            "Dyad",
            "NCP_score",
            "Predicted_occupancy"
        ]
    ]
    .reset_index(drop=True)
)


print(
    "Native nucleosomes entering Figure 3.16b:",
    len(
        figure_3_16b_scatter
    )
)


# ================================================================
# 6. Pearson correlation
# ================================================================
#
# Thesis reports approximately:
#
#     r = 0.23
#
# This calculation provides an important reconstruction check.
# ================================================================

figure_3_16b_pearson_r, \
figure_3_16b_pearson_p = pearsonr(

    figure_3_16b_scatter[
        "NCP_score"
    ],

    figure_3_16b_scatter[
        "Predicted_occupancy"
    ]
)


print()
print(
    "Pearson r:",
    figure_3_16b_pearson_r
)


print(
    "Pearson p:",
    figure_3_16b_pearson_p
)


# ================================================================
# 7. Prepare scatter-plot data
# ================================================================
#
# LEFT PANEL:
#
# x = NCP_score
# y = Predicted_occupancy
#
# figure_3_16b_scatter already contains the required data.
# ================================================================


# ================================================================
# 8. Binned analysis
# ================================================================
#
# The right panel of Figure 3.16b shows the relationship after
# binning nucleosomes by NCP score.
#
# The historical genome-wide code commonly used 30 bins for
# binned relationships.
#
# We therefore retain 30 equal-width NCP-score bins here.
#
# If the original Figure 3.16b code establishes a different bin
# number or binning convention, only this section needs changing.
# ================================================================

N_BINS = 30


ncp_values = (
    figure_3_16b_scatter[
        "NCP_score"
    ]
    .to_numpy(
        dtype=float
    )
)


predicted_values = (
    figure_3_16b_scatter[
        "Predicted_occupancy"
    ]
    .to_numpy(
        dtype=float
    )
)


# Equal-width bins spanning the observed NCP-score range.

bin_edges = np.linspace(

    np.min(
        ncp_values
    ),

    np.max(
        ncp_values
    ),

    N_BINS + 1
)


# np.digitize returns:
#
#     1 ... N_BINS
#
# for the bins defined above.

bin_numbers = np.digitize(
    ncp_values,
    bin_edges
)


# A value exactly equal to the maximum can fall into N_BINS + 1.
# Put that value into the final bin.

bin_numbers[
    bin_numbers
    ==
    N_BINS + 1
] = N_BINS


binned_rows = []


for bin_number in range(
    1,
    N_BINS + 1
):

    mask = (
        bin_numbers
        ==
        bin_number
    )


    n = int(
        np.sum(mask)
    )


    if n == 0:

        continue


    ncp_in_bin = (
        ncp_values[
            mask
        ]
    )


    predicted_in_bin = (
        predicted_values[
            mask
        ]
    )


    # ------------------------------------------------------------
    # Mean NCP score
    # ------------------------------------------------------------

    mean_ncp = np.mean(
        ncp_in_bin
    )


    # ------------------------------------------------------------
    # Mean predicted occupancy
    # ------------------------------------------------------------

    mean_predicted = np.mean(
        predicted_in_bin
    )


    # ------------------------------------------------------------
    # SEM
    # ------------------------------------------------------------

    if n > 1:

        sem_predicted = (
            np.std(
                predicted_in_bin,
                ddof=1
            )
            /
            np.sqrt(n)
        )

    else:

        sem_predicted = np.nan


    binned_rows.append(
        {
            "bin":
                bin_number,

            "bin_left":
                bin_edges[
                    bin_number - 1
                ],

            "bin_right":
                bin_edges[
                    bin_number
                ],

            "mean_NCP_score":
                mean_ncp,

            "mean_predicted_occupancy":
                mean_predicted,

            "sem_predicted_occupancy":
                sem_predicted,

            "n":
                n
        }
    )


figure_3_16b_binned = pd.DataFrame(
    binned_rows
)


# ================================================================
# 9. Add error-bar limits
# ================================================================

figure_3_16b_binned[
    "predicted_lower"
] = (

    figure_3_16b_binned[
        "mean_predicted_occupancy"
    ]

    -

    figure_3_16b_binned[
        "sem_predicted_occupancy"
    ]
)


figure_3_16b_binned[
    "predicted_upper"
] = (

    figure_3_16b_binned[
        "mean_predicted_occupancy"
    ]

    +

    figure_3_16b_binned[
        "sem_predicted_occupancy"
    ]
)


# ================================================================
# 10. Optional binned correlation
# ================================================================
#
# This is useful as a descriptive check but is NOT a replacement
# for the individual-nucleosome Pearson correlation above.
# ================================================================

if len(
    figure_3_16b_binned
) >= 2:

    figure_3_16b_binned_r, \
    figure_3_16b_binned_p = pearsonr(

        figure_3_16b_binned[
            "mean_NCP_score"
        ],

        figure_3_16b_binned[
            "mean_predicted_occupancy"
        ]
    )

else:

    figure_3_16b_binned_r = np.nan

    figure_3_16b_binned_p = np.nan


# ================================================================
# 11. Final summary
# ================================================================

print()
print(
    "Figure 3.16b ready."
)


print()
print(
    "Individual nucleosomes:",
    len(
        figure_3_16b_scatter
    )
)


print(
    "Number of NCP bins:",
    len(
        figure_3_16b_binned
    )
)


print()
print(
    "Individual-nucleosome Pearson r:",
    figure_3_16b_pearson_r
)


print(
    "Binned Pearson r:",
    figure_3_16b_binned_r
)


print()
print(
    figure_3_16b_binned
)