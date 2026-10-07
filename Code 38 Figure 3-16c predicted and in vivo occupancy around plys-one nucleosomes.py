

"""
SCRIPT NAME:
Code 38 Figure 3-16c predicted and in vivo occupancy around plus-one nucleosomes.py


WHAT THIS SCRIPT DOES:

Reconstructs the TOP PANEL ONLY of Figure 3.16c.

For each annotated yeast transcript:

1. Uses Pugh.txt to define:
       chromosome
       strand
       Experiment_Left
       Experiment_Right

2. Finds all independently mapped native nucleosome dyads from
   the SacCer3 native nucleosome map that fall within that
   transcript interval.

3. Orders those nucleosomes in transcriptional orientation.

4. Defines the first nucleosome in transcriptional order as the
   +1 nucleosome.

5. Extracts around that +1 dyad:

       a) predicted nucleosome occupancy from the
          10-mer cumulative-score + hard-rod model

       b) experimentally measured in-vivo nucleosome occupancy
          from Henikoff et al.

6. Reverses profiles for minus-strand genes so that:

       negative position = promoter / upstream
       positive position = gene body / downstream

7. Calculates the mean and SEM across genes.


NO INTRINSIC CYCLISABILITY IS CALCULATED HERE.

This script therefore reconstructs only the TOP panel of
Figure 3.16c.


THESIS SECTION / FIGURE:

Figure 3.16c, TOP PANEL ONLY


VARIABLES THAT MUST ALREADY EXIST:

yeast_pred_occ

    Dictionary produced by:

        Code 34 Figure 3-15 genome-wide nucleosome occupancy.py

    containing predicted occupancy:

        yeast_pred_occ["chr1"]
        ...
        yeast_pred_occ["chr16"]


henikoff_occ

    Dictionary containing experimentally measured in-vivo
    nucleosome occupancy from Henikoff et al.:

        henikoff_occ["chr1"]
        ...
        henikoff_occ["chr16"]

    Each value must be a NumPy-compatible array with one value
    per genomic base.

    The arrays must already be in SacCer3 coordinates.


INPUT FILES:

Pugh.txt

    Used ONLY for:

        Chrom
        Strand
        Experiment_Left
        Experiment_Right

    IMPORTANT:

    Pugh's PlusOne_Dyad column is NOT used.


Saccer3bed.bed

    Native yeast nucleosome map after SacCer2 -> SacCer3 liftOver.

    The dyad of each mapped nucleosome is:

        Start + 10


OUTPUT VARIABLES:

df6

    Transcript table with transcriptionally ordered native
    nucleosome centres.


plus1_table

    Transcripts with at least one mapped native nucleosome.


predicted_plus1_profiles_316c

    Individual predicted occupancy profiles around +1.


measured_plus1_profiles_316c

    Individual in-vivo Henikoff occupancy profiles around +1.


figure_3_16c_top

    Mean +/- SEM data for the top panel of Figure 3.16c.


IMPORTANT:

The +1 nucleosome is NOT taken from Pugh's PlusOne_Dyad.

Instead:

    + strand:
        first / leftmost mapped native nucleosome in the
        experimental transcript interval

    - strand:
        first / rightmost mapped native nucleosome in the
        experimental transcript interval

This is the same independently defined +1 procedure used for
Figure 3.15b.
"""


import numpy as np
import pandas as pd


# ================================================================
# 1. Input files
# ================================================================

PUGH_FILE = "Pugh.txt"

SACCER3_NUCLEOSOME_BED = "Saccer3bed.bed"


# ================================================================
# 2. Window around +1
# ================================================================
#
# Keep this explicit because the exact historical plotting range
# can be changed here without affecting the rest of the analysis.
# ================================================================

PLUS1_WINDOW = 1000


relative_positions = np.arange(
    -PLUS1_WINDOW,
    PLUS1_WINDOW + 1
)


# ================================================================
# 3. Check required occupancy variables
# ================================================================

if "yeast_pred_occ" not in globals():

    raise NameError(
        "yeast_pred_occ is not defined. "
        "Run the genome-wide 10-mer + hard-rod prediction first."
    )


if "henikoff_occ" not in globals():

    raise NameError(
        "henikoff_occ is not defined. "
        "Load the Henikoff in-vivo nucleosome occupancy data "
        "into chromosome-length arrays before running this script."
    )


# ================================================================
# 4. Chromosome conversion
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


def convert_chromosome(chromosome):

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

        chromosome_part = chromosome


    # Already numeric.

    if chromosome_part.isdigit():

        return (
            "chr"
            +
            str(
                int(
                    chromosome_part
                )
            )
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


    return (
        "chr"
        +
        str(
            ROMAN_TO_ARABIC[
                chromosome_part
            ]
        )
    )


# ================================================================
# 5. Load independently mapped native nucleosomes
# ================================================================
#
# This is the SacCer3 liftOver product.
#
# The BED interval was constructed around the original dyad as:
#
#     dyad - 10
#     dyad + 10
#
# Therefore the mapped SacCer3 dyad is reconstructed as:
#
#     Start + 10
# ================================================================

df3 = pd.read_csv(
    SACCER3_NUCLEOSOME_BED,
    sep="\t",
    header=None,
    names=[
        "Chromosome",
        "Start",
        "End"
    ]
)


df3["Dyad"] = (
    df3["Start"]
    +
    10
)


df3 = df3[
    [
        "Chromosome",
        "Dyad"
    ]
].copy()


df3["Chromosome"] = (
    df3["Chromosome"]
    .apply(
        convert_chromosome
    )
)


df3["Dyad"] = pd.to_numeric(
    df3["Dyad"],
    errors="coerce"
)


df3 = df3[
    df3["Dyad"].notna()
].copy()


df3["Dyad"] = (
    df3["Dyad"]
    .astype(int)
)


# Explicitly put native nucleosomes into increasing genomic
# coordinate order.

df3 = (
    df3
    .sort_values(
        [
            "Chromosome",
            "Dyad"
        ]
    )
    .reset_index(drop=True)
)


print(
    "Mapped native nucleosomes:",
    len(df3)
)


# ================================================================
# 6. Load Pugh transcript annotation
# ================================================================

df4 = pd.read_csv(
    PUGH_FILE,
    sep="\t"
)


required_columns = {
    "Systematic ID",
    "Feature class Level 1",
    "Chrom",
    "Strand",
    "Experiment_Left",
    "Experiment_Right"
}


missing_columns = (
    required_columns
    -
    set(df4.columns)
)


if missing_columns:

    raise ValueError(
        "Pugh.txt is missing: "
        +
        ", ".join(
            sorted(missing_columns)
        )
    )


# ================================================================
# 7. Keep only the columns actually used
# ================================================================
#
# PlusOne_Dyad is deliberately excluded.
# ================================================================

df5 = df4[
    [
        "Systematic ID",
        "Feature class Level 1",
        "Chrom",
        "Strand",
        "Experiment_Left",
        "Experiment_Right"
    ]
].copy()


# ================================================================
# 8. Construct df6
# ================================================================
#
# We are NOT restricting Feature class Level 1 to 01/02/03/04.
#
# As in the analysis discussed previously, all rows with a
# non-missing Feature class Level 1 are considered.
# ================================================================

df6 = df5[
    ~df5[
        "Feature class Level 1"
    ].isna()
].copy()


df6 = df6[
    df6["Chrom"].notna()
    &
    df6["Strand"].notna()
    &
    df6["Experiment_Left"].notna()
    &
    df6["Experiment_Right"].notna()
].copy()


df6 = df6[
    df6["Strand"].isin(
        [
            "+",
            "-"
        ]
    )
].copy()


# ================================================================
# 9. Standardise chromosome names
# ================================================================

df6["Chrom"] = (
    df6["Chrom"]
    .apply(
        convert_chromosome
    )
)


# ================================================================
# 10. Convert transcript coordinates
# ================================================================

df6["Experiment_Left"] = (
    pd.to_numeric(
        df6["Experiment_Left"],
        errors="coerce"
    )
)


df6["Experiment_Right"] = (
    pd.to_numeric(
        df6["Experiment_Right"],
        errors="coerce"
    )
)


df6 = df6[
    df6["Experiment_Left"].notna()
    &
    df6["Experiment_Right"].notna()
].copy()


df6["Experiment_Left"] = (
    df6["Experiment_Left"]
    .astype(int)
)


df6["Experiment_Right"] = (
    df6["Experiment_Right"]
    .astype(int)
)


# ================================================================
# 11. Find mapped native nucleosomes inside each transcript
# ================================================================

def get_nucleosome_centers(row):

    chrom = row[
        "Chrom"
    ]

    left = row[
        "Experiment_Left"
    ]

    right = row[
        "Experiment_Right"
    ]


    centers = (
        df3.loc[
            (
                df3["Chromosome"]
                ==
                chrom
            )
            &
            (
                df3["Dyad"]
                >=
                left
            )
            &
            (
                df3["Dyad"]
                <=
                right
            ),
            "Dyad"
        ]
        .tolist()
    )


    return centers


df6[
    "Nucleosome centers"
] = df6.apply(
    get_nucleosome_centers,
    axis=1
)


# ================================================================
# 12. Put nucleosome centers into transcriptional orientation
# ================================================================
#
# For plus-strand genes:
#
#     left -> right
#
# already corresponds to transcriptional order.
#
#
# For minus-strand genes:
#
#     right -> left
#
# is transcriptional order.
#
# Therefore reverse the list.
# ================================================================

minus_mask = (
    df6["Strand"]
    ==
    "-"
)


df6.loc[
    minus_mask,
    "Nucleosome centers"
] = (
    df6.loc[
        minus_mask,
        "Nucleosome centers"
    ]
    .apply(
        lambda x: x[::-1]
    )
)


# ================================================================
# 13. Independently define +1 nucleosome
# ================================================================
#
# The first native nucleosome in transcriptional order is +1.
# ================================================================

df6[
    "Number of nucleosomes"
] = (
    df6[
        "Nucleosome centers"
    ]
    .apply(len)
)


plus1_table = df6[
    df6[
        "Number of nucleosomes"
    ]
    >
    0
].copy()


plus1_table[
    "Plus1_Dyad_from_native_map"
] = (
    plus1_table[
        "Nucleosome centers"
    ]
    .apply(
        lambda x: x[0]
    )
)


plus1_table[
    "Plus1_Dyad_from_native_map"
] = (
    plus1_table[
        "Plus1_Dyad_from_native_map"
    ]
    .astype(int)
)


print()
print(
    "Transcripts considered:",
    len(df6)
)


print(
    "Transcripts with independently mapped +1 nucleosome:",
    len(plus1_table)
)


# ================================================================
# 14. Check chromosome dictionaries
# ================================================================

expected_chromosomes = {
    f"chr{i}"
    for i in range(
        1,
        17
    )
}


missing_predicted = (
    expected_chromosomes
    -
    set(
        yeast_pred_occ.keys()
    )
)


missing_henikoff = (
    expected_chromosomes
    -
    set(
        henikoff_occ.keys()
    )
)


if missing_predicted:

    raise ValueError(
        "yeast_pred_occ is missing chromosomes: "
        +
        ", ".join(
            sorted(
                missing_predicted
            )
        )
    )


if missing_henikoff:

    raise ValueError(
        "henikoff_occ is missing chromosomes: "
        +
        ", ".join(
            sorted(
                missing_henikoff
            )
        )
    )


# ================================================================
# 15. Extract predicted and measured occupancy around +1
# ================================================================

predicted_plus1_profiles_316c = []

measured_plus1_profiles_316c = []

plus1_metadata_316c = []


expected_profile_length = (
    2 * PLUS1_WINDOW
    +
    1
)


for row_index, row in (
    plus1_table.iterrows()
):

    chrom = row[
        "Chrom"
    ]

    strand = row[
        "Strand"
    ]

    dyad = int(
        row[
            "Plus1_Dyad_from_native_map"
        ]
    )


    # ------------------------------------------------------------
    # Retrieve chromosome arrays
    # ------------------------------------------------------------

    predicted_chrom = np.asarray(
        yeast_pred_occ[
            chrom
        ],
        dtype=float
    )


    measured_chrom = np.asarray(
        henikoff_occ[
            chrom
        ],
        dtype=float
    )


    # ------------------------------------------------------------
    # These should both represent the same SacCer3 chromosome.
    # ------------------------------------------------------------

    if (
        len(predicted_chrom)
        !=
        len(measured_chrom)
    ):

        raise ValueError(
            f"Chromosome-length mismatch for {chrom}: "
            f"predicted={len(predicted_chrom)}, "
            f"Henikoff={len(measured_chrom)}"
        )


    # ------------------------------------------------------------
    # Window around independently identified +1 dyad
    # ------------------------------------------------------------

    start = (
        dyad
        -
        PLUS1_WINDOW
    )


    end = (
        dyad
        +
        PLUS1_WINDOW
        +
        1
    )


    # Skip genes too close to chromosome ends.

    if start < 0:

        continue


    if end > len(
        predicted_chrom
    ):

        continue


    # ------------------------------------------------------------
    # Extract predicted occupancy
    # ------------------------------------------------------------

    predicted_profile = (
        predicted_chrom[
            start:end
        ]
        .copy()
    )


    # ------------------------------------------------------------
    # Extract measured in-vivo occupancy
    # ------------------------------------------------------------

    measured_profile = (
        measured_chrom[
            start:end
        ]
        .copy()
    )


    if (
        len(predicted_profile)
        !=
        expected_profile_length
    ):

        continue


    if (
        len(measured_profile)
        !=
        expected_profile_length
    ):

        continue


    # ============================================================
    # 16. Orient profiles in direction of transcription
    # ============================================================
    #
    # After this transformation:
    #
    #     negative x = promoter / upstream
    #     zero       = +1 dyad
    #     positive x = gene body / downstream
    #
    # Plus-strand genes require no change.
    #
    # Minus-strand genes are reversed.
    # ============================================================

    if strand == "-":

        predicted_profile = (
            predicted_profile[
                ::-1
            ]
        )


        measured_profile = (
            measured_profile[
                ::-1
            ]
        )


    # ------------------------------------------------------------
    # Store profiles
    # ------------------------------------------------------------

    predicted_plus1_profiles_316c.append(
        predicted_profile
    )


    measured_plus1_profiles_316c.append(
        measured_profile
    )


    plus1_metadata_316c.append(
        {
            "df6_index":
                row_index,

            "Systematic ID":
                row[
                    "Systematic ID"
                ],

            "Chrom":
                chrom,

            "Strand":
                strand,

            "Plus1_Dyad":
                dyad,

            "Number of nucleosomes":
                row[
                    "Number of nucleosomes"
                ]
        }
    )


# ================================================================
# 17. Convert to matrices
# ================================================================

predicted_plus1_profiles_316c = (
    np.asarray(
        predicted_plus1_profiles_316c,
        dtype=float
    )
)


measured_plus1_profiles_316c = (
    np.asarray(
        measured_plus1_profiles_316c,
        dtype=float
    )
)


plus1_metadata_316c = (
    pd.DataFrame(
        plus1_metadata_316c
    )
)


# ================================================================
# 18. Sanity checks
# ================================================================

if (
    len(
        predicted_plus1_profiles_316c
    )
    ==
    0
):

    raise ValueError(
        "No usable +1 profiles were obtained."
    )


assert (
    predicted_plus1_profiles_316c.shape
    ==
    measured_plus1_profiles_316c.shape
)


assert (
    predicted_plus1_profiles_316c.shape[1]
    ==
    expected_profile_length
)


assert (
    predicted_plus1_profiles_316c.shape[0]
    ==
    len(
        plus1_metadata_316c
    )
)


print()
print(
    "Profiles entering Figure 3.16c:",
    predicted_plus1_profiles_316c.shape
)


# ================================================================
# 19. Mean occupancy profiles
# ================================================================

predicted_mean_316c = np.nanmean(
    predicted_plus1_profiles_316c,
    axis=0
)


measured_mean_316c = np.nanmean(
    measured_plus1_profiles_316c,
    axis=0
)


# ================================================================
# 20. SEM
# ================================================================

def calculate_nansem(
    matrix
):

    n = np.sum(
        np.isfinite(
            matrix
        ),
        axis=0
    )


    sd = np.nanstd(
        matrix,
        axis=0,
        ddof=1
    )


    sem = (
        sd
        /
        np.sqrt(n)
    )


    return sem, n


predicted_sem_316c, \
predicted_n_316c = (
    calculate_nansem(
        predicted_plus1_profiles_316c
    )
)


measured_sem_316c, \
measured_n_316c = (
    calculate_nansem(
        measured_plus1_profiles_316c
    )
)


# ================================================================
# 21. Assemble Figure 3.16c TOP PANEL data
# ================================================================

figure_3_16c_top = pd.DataFrame(
    {
        "position_relative_to_plus1_dyad":
            relative_positions,

        "predicted_mean":
            predicted_mean_316c,

        "predicted_sem":
            predicted_sem_316c,

        "predicted_n":
            predicted_n_316c,

        "measured_in_vivo_mean":
            measured_mean_316c,

        "measured_in_vivo_sem":
            measured_sem_316c,

        "measured_in_vivo_n":
            measured_n_316c
    }
)


# ================================================================
# 22. SEM-band boundaries
# ================================================================

figure_3_16c_top[
    "predicted_lower"
] = (

    figure_3_16c_top[
        "predicted_mean"
    ]

    -

    figure_3_16c_top[
        "predicted_sem"
    ]
)


figure_3_16c_top[
    "predicted_upper"
] = (

    figure_3_16c_top[
        "predicted_mean"
    ]

    +

    figure_3_16c_top[
        "predicted_sem"
    ]
)


figure_3_16c_top[
    "measured_lower"
] = (

    figure_3_16c_top[
        "measured_in_vivo_mean"
    ]

    -

    figure_3_16c_top[
        "measured_in_vivo_sem"
    ]
)


figure_3_16c_top[
    "measured_upper"
] = (

    figure_3_16c_top[
        "measured_in_vivo_mean"
    ]

    +

    figure_3_16c_top[
        "measured_in_vivo_sem"
    ]
)


# ================================================================
# 23. Final checks
# ================================================================

assert (
    len(
        figure_3_16c_top
    )
    ==
    expected_profile_length
)


assert (
    figure_3_16c_top[
        "position_relative_to_plus1_dyad"
    ].iloc[
        PLUS1_WINDOW
    ]
    ==
    0
)


print()
print(
    "Figure 3.16c TOP PANEL data ready."
)


print(
    "Number of genes:",
    len(
        plus1_metadata_316c
    )
)


print(
    "Relative coordinate range:",
    relative_positions[0],
    "to",
    relative_positions[-1],
    "bp"
)


print()
print(
    figure_3_16c_top.head()
)