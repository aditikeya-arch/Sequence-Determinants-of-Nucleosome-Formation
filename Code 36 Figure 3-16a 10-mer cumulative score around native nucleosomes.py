

"""
SCRIPT NAME:
Code 36 Figure 3-16a 10-mer cumulative score around native nucleosomes.py


WHAT THIS SCRIPT DOES:

Reconstructs Figure 3.16a.

The 10-mer cumulative score is calculated genome-wide at
single-base-pair resolution using a centred 108-bp window.

The mean 10-mer cumulative-score profile is then calculated
around:

    1. all experimentally mapped native yeast nucleosome dyads

    2. an equal number of randomly selected genomic coordinates

The output is the data required to plot mean 10-mer cumulative
score as a function of distance from the reference coordinate.

No plotting is performed.


THESIS SECTION / FIGURE:

Figure 3.16a


FIGURE DEFINITION:

The thesis describes this panel as:

    mean 10-mer cumulative score vs distance from the dyad,
    compared with randomly chosen genomic coordinates,
    averaged over 67,539 mapped yeast nucleosomes.


INPUT FILE:

    Saccer3bed.bed

This is the SacCer3 version of the native yeast nucleosome map.

It was generated from:

    NucleosomeSacCer2.txt

by constructing +/- 10 bp BED intervals around the original
SacCer2 dyads and converting them to SacCer3 using UCSC liftOver:

    sacCer2ToSacCer3.over.chain.gz


VARIABLES THAT MUST ALREADY EXIST:

sequences_data

    List containing the 16 SacCer3 chromosome sequences:

        sequences_data[0]  = chromosome 1
        ...
        sequences_data[15] = chromosome 16


df_enrichment_rc

    Reverse-complement-collapsed 10-mer enrichment table generated
    in:

        Code 20 Figure 2-25a 10-mer enrichment distribution.py

    Required columns:

        kmer
        log2_enrichment


canonical_kmer

    Function defined in Code 20:

        canonical_kmer(kmer)

    Returns the lexicographically canonical member of a 10-mer /
    reverse-complement pair.


OTHER SCRIPTS THAT MUST BE RUN FIRST:

Code 20 Figure 2-25a 10-mer enrichment distribution.py


OUTPUT VARIABLES:

yeast_10mer_CS

    Dictionary containing the genome-wide 108-bp cumulative score:

        yeast_10mer_CS["chr1"]
        ...
        yeast_10mer_CS["chr16"]

    Each array has the same length as the corresponding chromosome.


native_dyad_profiles

    Matrix of 10-mer cumulative-score profiles around native
    nucleosome dyads.


random_coordinate_profiles

    Equivalent matrix around random genomic coordinates.


figure_3_16a

    Dataframe containing the mean and SEM profiles required for
    Figure 3.16a.


NOTES:

The cumulative score uses a 108-bp sequence window.

Each 108-bp window contains:

    108 - 10 + 1 = 99

overlapping 10-mers.

The score is assigned to:

    window_start + 54

as in the genome-wide calculation used previously.

The native nucleosome coordinates come from the SacCer3 liftOver
map and are therefore used directly as 0-based genomic positions.
"""


import numpy as np
import pandas as pd


# ================================================================
# 1. Constants
# ================================================================

SACCER3_NUCLEOSOME_BED = "Saccer3bed.bed"

K = 10

SCORE_WINDOW = 108

ASSIGNED_POSITION = 54


# Distance shown around each nucleosome dyad.
#
# Change this only if the exact plotting range of the historical
# figure needs to be reproduced differently.

PROFILE_WINDOW = 500


relative_positions = np.arange(
    -PROFILE_WINDOW,
    PROFILE_WINDOW + 1
)


# ================================================================
# 2. Chromosome-name conversion
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


def convert_roman_to_arabic(chromosome):

    chromosome = str(chromosome).strip()

    if chromosome.lower().startswith("chr"):

        chromosome_part = chromosome[3:]

    else:

        chromosome_part = chromosome


    if chromosome_part.isdigit():

        return (
            "chr"
            +
            str(int(chromosome_part))
        )


    chromosome_part = chromosome_part.upper()


    if chromosome_part not in ROMAN_TO_ARABIC:

        raise ValueError(
            "Unrecognised chromosome: "
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
# 3. Load SacCer3 native nucleosome coordinates
# ================================================================
#
# The original SacCer2 BED intervals were constructed as:
#
#     Start = dyad - 10
#     End   = dyad + 10
#
# After liftOver to SacCer3, the dyad is reconstructed as:
#
#     Dyad = Start + 10
#
# This follows the same construction used in the native
# nucleosome analysis.
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
        convert_roman_to_arabic
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
    "Number of mapped SacCer3 nucleosomes:",
    len(df3)
)


# The thesis reports 67,539 mapped nucleosomes.
#
# Keep this as a check rather than silently changing the dataset.

if len(df3) != 67539:

    print(
        "NOTE: thesis reports 67,539 mapped nucleosomes, "
        f"whereas df3 currently contains {len(df3):,}."
    )


# ================================================================
# 4. Construct the 10-mer score dictionary
# ================================================================
#
# Code 20 contains one row per reverse-complement-collapsed
# 10-mer class.
# ================================================================

required_columns = {
    "kmer",
    "log2_enrichment"
}


missing_columns = (
    required_columns
    -
    set(df_enrichment_rc.columns)
)


if missing_columns:

    raise ValueError(
        "df_enrichment_rc is missing: "
        +
        ", ".join(
            sorted(missing_columns)
        )
    )


kmer_score_dict = dict(
    zip(
        df_enrichment_rc["kmer"],
        df_enrichment_rc["log2_enrichment"]
    )
)


print(
    "Number of reverse-complement-collapsed "
    "10-mer classes:",
    len(kmer_score_dict)
)


# ================================================================
# 5. Calculate genome-wide 108-bp cumulative score
# ================================================================
#
# For every 108-bp genomic window:
#
#     99 overlapping 10-mers
#
# are scored.
#
# The cumulative score is:
#
#     sum(log2 enrichment of each 10-mer)
#
# The value is assigned to:
#
#     window_start + 54
#
# giving a single-base-resolution genome-wide score.
#
# If any 10-mer in the window has no enrichment score, that
# position is left as NaN.
# ================================================================

def compute_chromosome_10mer_CS(
    chromosome_sequence,
    kmer_score_dict,
    k=10,
    window=108,
    assigned_position=54
):

    sequence = (
        chromosome_sequence
        .upper()
    )

    chromosome_length = len(sequence)


    output = np.full(
        chromosome_length,
        np.nan,
        dtype=float
    )


    number_of_kmers = (
        window
        -
        k
        +
        1
    )


    # ------------------------------------------------------------
    # First obtain the score of every individual genomic 10-mer.
    # ------------------------------------------------------------

    individual_kmer_scores = np.full(
        chromosome_length - k + 1,
        np.nan,
        dtype=float
    )


    for i in range(
        chromosome_length - k + 1
    ):

        kmer = sequence[
            i:i + k
        ]


        # Ignore windows containing ambiguous bases.

        if (
            len(kmer) != k
            or
            any(
                base not in "ACGT"
                for base in kmer
            )
        ):

            continue


        canonical = canonical_kmer(
            kmer
        )


        score = kmer_score_dict.get(
            canonical,
            np.nan
        )


        individual_kmer_scores[i] = score


    # ------------------------------------------------------------
    # Rolling 99-kmer sum.
    #
    # A 108-bp window contains exactly 99 overlapping 10-mers.
    # ------------------------------------------------------------

    finite = np.isfinite(
        individual_kmer_scores
    )


    values = np.where(
        finite,
        individual_kmer_scores,
        0.0
    )


    cumulative_values = np.concatenate(
        (
            [0.0],
            np.cumsum(values)
        )
    )


    cumulative_valid = np.concatenate(
        (
            [0],
            np.cumsum(
                finite.astype(int)
            )
        )
    )


    number_of_windows = (
        chromosome_length
        -
        window
        +
        1
    )


    for start in range(
        number_of_windows
    ):

        end_kmer = (
            start
            +
            number_of_kmers
        )


        n_valid = (
            cumulative_valid[end_kmer]
            -
            cumulative_valid[start]
        )


        # Require all 99 component 10-mers.

        if n_valid != number_of_kmers:

            continue


        score_sum = (
            cumulative_values[end_kmer]
            -
            cumulative_values[start]
        )


        genomic_position = (
            start
            +
            assigned_position
        )


        output[
            genomic_position
        ] = score_sum


    return output


# ================================================================
# 6. Calculate the score for all 16 chromosomes
# ================================================================

if len(sequences_data) != 16:

    raise ValueError(
        "sequences_data must contain exactly 16 "
        "SacCer3 chromosome sequences."
    )


yeast_10mer_CS = {}


for chromosome_number in range(
    1,
    17
):

    chrom = (
        f"chr{chromosome_number}"
    )


    print(
        "Calculating 10-mer cumulative score for",
        chrom
    )


    yeast_10mer_CS[chrom] = (
        compute_chromosome_10mer_CS(
            sequences_data[
                chromosome_number - 1
            ],
            kmer_score_dict,
            k=K,
            window=SCORE_WINDOW,
            assigned_position=ASSIGNED_POSITION
        )
    )


# ================================================================
# 7. Extract profiles around native nucleosome dyads
# ================================================================

native_dyad_profiles = []

native_dyad_metadata = []


for row_index, row in df3.iterrows():

    chrom = row["Chromosome"]

    dyad = int(
        row["Dyad"]
    )


    chromosome_scores = (
        yeast_10mer_CS[
            chrom
        ]
    )


    start = (
        dyad
        -
        PROFILE_WINDOW
    )


    end = (
        dyad
        +
        PROFILE_WINDOW
        +
        1
    )


    # Skip nucleosomes too close to chromosome ends.

    if start < 0:
        continue


    if end > len(
        chromosome_scores
    ):
        continue


    profile = (
        chromosome_scores[
            start:end
        ]
        .copy()
    )


    if len(profile) != len(
        relative_positions
    ):
        continue


    native_dyad_profiles.append(
        profile
    )


    native_dyad_metadata.append(
        {
            "df3_index":
                row_index,

            "Chromosome":
                chrom,

            "Dyad":
                dyad
        }
    )


native_dyad_profiles = np.asarray(
    native_dyad_profiles,
    dtype=float
)


native_dyad_metadata = pd.DataFrame(
    native_dyad_metadata
)


print()
print(
    "Native dyad profiles:",
    native_dyad_profiles.shape
)


# ================================================================
# 8. Random genomic-coordinate control
# ================================================================
#
# The thesis compares the mapped nucleosomes with randomly chosen
# genomic coordinates.
#
# We draw the SAME NUMBER of random coordinates as the number of
# mapped nucleosomes.
#
# Chromosomes are sampled in proportion to their lengths, which
# is equivalent to drawing uniformly from the concatenated yeast
# genome.
#
# No random seed is imposed here unless the original analysis
# explicitly used one.
# ================================================================

N_RANDOM = len(df3)


chromosome_names = [
    f"chr{i}"
    for i in range(1, 17)
]


chromosome_lengths = np.array(
    [
        len(
            sequences_data[i - 1]
        )
        for i in range(
            1,
            17
        )
    ],
    dtype=int
)


# Only positions with enough sequence on both sides are eligible.

valid_lengths = (
    chromosome_lengths
    -
    2 * PROFILE_WINDOW
)


if np.any(
    valid_lengths <= 0
):

    raise ValueError(
        "PROFILE_WINDOW is too large for at least "
        "one chromosome."
    )


# Probability of choosing each chromosome.

chromosome_probabilities = (
    valid_lengths
    /
    valid_lengths.sum()
)


random_chromosomes = np.random.choice(
    chromosome_names,
    size=N_RANDOM,
    replace=True,
    p=chromosome_probabilities
)


random_coordinates = np.empty(
    N_RANDOM,
    dtype=int
)


for i, chrom in enumerate(
    random_chromosomes
):

    chromosome_number = int(
        chrom.replace(
            "chr",
            ""
        )
    )


    chromosome_length = (
        chromosome_lengths[
            chromosome_number - 1
        ]
    )


    # randint upper boundary is exclusive.
    #
    # The resulting coordinate always has PROFILE_WINDOW bases
    # available on either side.

    random_coordinates[i] = (
        np.random.randint(
            PROFILE_WINDOW,
            chromosome_length
            -
            PROFILE_WINDOW
        )
    )


# ================================================================
# 9. Extract random-coordinate profiles
# ================================================================

random_coordinate_profiles = []

random_coordinate_metadata = []


for chrom, coordinate in zip(
    random_chromosomes,
    random_coordinates
):

    chromosome_scores = (
        yeast_10mer_CS[
            chrom
        ]
    )


    start = (
        coordinate
        -
        PROFILE_WINDOW
    )


    end = (
        coordinate
        +
        PROFILE_WINDOW
        +
        1
    )


    profile = (
        chromosome_scores[
            start:end
        ]
        .copy()
    )


    if len(profile) != len(
        relative_positions
    ):
        continue


    random_coordinate_profiles.append(
        profile
    )


    random_coordinate_metadata.append(
        {
            "Chromosome":
                chrom,

            "Coordinate":
                coordinate
        }
    )


random_coordinate_profiles = np.asarray(
    random_coordinate_profiles,
    dtype=float
)


random_coordinate_metadata = pd.DataFrame(
    random_coordinate_metadata
)


print(
    "Random-coordinate profiles:",
    random_coordinate_profiles.shape
)


# ================================================================
# 10. Calculate mean profiles
# ================================================================

native_mean = np.nanmean(
    native_dyad_profiles,
    axis=0
)


random_mean = np.nanmean(
    random_coordinate_profiles,
    axis=0
)


# ================================================================
# 11. Calculate SEM
# ================================================================

def calculate_nansem(matrix):

    n = np.sum(
        np.isfinite(matrix),
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


native_sem, native_n = (
    calculate_nansem(
        native_dyad_profiles
    )
)


random_sem, random_n = (
    calculate_nansem(
        random_coordinate_profiles
    )
)


# ================================================================
# 12. Assemble Figure 3.16a data
# ================================================================

figure_3_16a = pd.DataFrame(
    {
        "position_relative_to_reference":
            relative_positions,

        "native_nucleosome_mean":
            native_mean,

        "native_nucleosome_sem":
            native_sem,

        "native_nucleosome_n":
            native_n,

        "random_coordinate_mean":
            random_mean,

        "random_coordinate_sem":
            random_sem,

        "random_coordinate_n":
            random_n
    }
)


# ================================================================
# 13. Mean +/- SEM boundaries
# ================================================================

figure_3_16a[
    "native_lower"
] = (
    figure_3_16a[
        "native_nucleosome_mean"
    ]
    -
    figure_3_16a[
        "native_nucleosome_sem"
    ]
)


figure_3_16a[
    "native_upper"
] = (
    figure_3_16a[
        "native_nucleosome_mean"
    ]
    +
    figure_3_16a[
        "native_nucleosome_sem"
    ]
)


figure_3_16a[
    "random_lower"
] = (
    figure_3_16a[
        "random_coordinate_mean"
    ]
    -
    figure_3_16a[
        "random_coordinate_sem"
    ]
)


figure_3_16a[
    "random_upper"
] = (
    figure_3_16a[
        "random_coordinate_mean"
    ]
    +
    figure_3_16a[
        "random_coordinate_sem"
    ]
)


# ================================================================
# 14. Final checks
# ================================================================

assert (
    figure_3_16a[
        "position_relative_to_reference"
    ].iloc[PROFILE_WINDOW]
    ==
    0
)


assert (
    len(figure_3_16a)
    ==
    2 * PROFILE_WINDOW
    +
    1
)


print()
print(
    "Figure 3.16a data ready."
)


print(
    "Native nucleosomes in df3:",
    len(df3)
)


print(
    "Native profiles used:",
    len(native_dyad_profiles)
)


print(
    "Random genomic coordinates generated:",
    len(random_coordinate_profiles)
)


print(
    "Relative-position range:",
    relative_positions[0],
    "to",
    relative_positions[-1],
    "bp"
)


print()
print(
    figure_3_16a.head()
)