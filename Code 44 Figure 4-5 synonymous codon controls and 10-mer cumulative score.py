

"""
SCRIPT NAME:
Code 44 Figure 4-6 synonymous codon controls and 10-mer cumulative score.py


WHAT THIS SCRIPT DOES:

Reconstructs Figure 4.6.

This repeats the synonymous-codon analyses used for Figure 4.5,
but replaces intrinsic cyclisability with the nucleosome-SELEX
10-mer cumulative enrichment score.


FIGURE 4.6a

Uses the SAME random-peptide reverse-translated DNA sequences
already generated for Figure 4.5a:

    random_peptide_spatial_sequences
    random_peptide_uniform_sequences

For each 297-bp sequence, calculates a rolling cumulative
10-mer score over every 108-bp window.


FIGURE 4.6b

Uses the SAME native and synonymous-control sequences generated
for Figure 4.5b:

    native_sequences_45b
    random_synonymous_sequences_45b
    overall_usage_sequences_45b
    spatial_usage_sequences_45b

Calculates rolling 108-bp cumulative 10-mer scores.

The Figure 4.6b profiles are then mean-centered separately
for each sequence before averaging.


10-MER SCORE DEFINITION:

The 10-mer score table comes from:

    Code 20 Figure 2-25a 10-mer enrichment distribution.py

For each 108-bp window:

    108 - 10 + 1 = 99

overlapping 10-mers are scored.

The cumulative score is:

    sum of the 99 log2 enrichment values

where enrichment is the reverse-complement-collapsed,
normalized-frequency R6 / R1 enrichment.


VARIABLES THAT MUST ALREADY EXIST:

From Code 20:

    df_enrichment_rc
    canonical_kmer


From Code 43 / Figure 4.5:

    random_peptide_spatial_sequences
    random_peptide_uniform_sequences

    native_sequences_45b
    random_synonymous_sequences_45b
    overall_usage_sequences_45b
    spatial_usage_sequences_45b


OUTPUT VARIABLES:

tenmer_46a_spatial
tenmer_46a_uniform

tenmer_46b_native
tenmer_46b_random
tenmer_46b_overall
tenmer_46b_spatial


figure_4_6a

    Mean rolling 10-mer cumulative-score profiles for:

        Spatially varying
        Uniform


figure_4_6b

    Mean-centered rolling 10-mer cumulative-score profiles for:

        Native
        Random synonymous
        Overall codon usage
        Spatially varying


NO PLOTTING IS PERFORMED.
"""


import numpy as np
import pandas as pd


# ================================================================
# 1. Check prerequisites
# ================================================================

required_variables = [

    "df_enrichment_rc",
    "canonical_kmer",

    "random_peptide_spatial_sequences",
    "random_peptide_uniform_sequences",

    "native_sequences_45b",
    "random_synonymous_sequences_45b",
    "overall_usage_sequences_45b",
    "spatial_usage_sequences_45b"
]


for variable_name in required_variables:

    if variable_name not in globals():

        raise NameError(
            f"{variable_name} is not defined."
        )


# ================================================================
# 2. Constants
# ================================================================

SEQUENCE_LENGTH = 297

K = 10

SCORE_WINDOW = 108


# Number of 108-bp windows in a 297-bp sequence.

N_SCORE_POSITIONS = (

    SEQUENCE_LENGTH
    -
    SCORE_WINDOW
    +
    1
)


assert N_SCORE_POSITIONS == 190


# Number of overlapping 10-mers in each 108-bp window.

N_KMERS_PER_WINDOW = (

    SCORE_WINDOW
    -
    K
    +
    1
)


assert N_KMERS_PER_WINDOW == 99


# ================================================================
# 3. Construct 10-mer score dictionary
# ================================================================
#
# Code 20 stores one row for each reverse-complement-collapsed
# 10-mer class.
#
# The representative sequence is in:
#
#     kmer
#
# and the score is:
#
#     log2_enrichment
# ================================================================

required_columns = {
    "kmer",
    "log2_enrichment"
}


missing_columns = (

    required_columns
    -
    set(
        df_enrichment_rc.columns
    )
)


if missing_columns:

    raise ValueError(
        "df_enrichment_rc is missing columns: "
        +
        ", ".join(
            sorted(
                missing_columns
            )
        )
    )


tenmer_score_dict = dict(

    zip(

        df_enrichment_rc[
            "kmer"
        ],

        df_enrichment_rc[
            "log2_enrichment"
        ]
    )
)


print(
    "10-mer reverse-complement classes:",
    len(
        tenmer_score_dict
    )
)


# ================================================================
# 4. Score one 297-bp sequence
# ================================================================
#
# First calculate the score of every individual overlapping
# 10-mer:
#
#     297 - 10 + 1 = 288
#
# Then use a rolling sum of 99 consecutive 10-mer scores.
#
# Each rolling sum corresponds to one 108-bp DNA window.
#
# This is much faster than repeatedly rescoring all 99 10-mers
# independently for every 108-bp window.
# ================================================================

def rolling_10mer_cumulative_score(
    sequence
):

    sequence = str(
        sequence
    ).upper()


    if len(
        sequence
    ) != SEQUENCE_LENGTH:

        raise ValueError(
            "Sequence must be exactly 297 bp."
        )


    # ------------------------------------------------------------
    # Score all 288 overlapping 10-mers.
    # ------------------------------------------------------------

    individual_scores = np.empty(

        SEQUENCE_LENGTH
        -
        K
        +
        1,

        dtype=float
    )


    for i in range(
        len(
            individual_scores
        )
    ):

        kmer = sequence[
            i:i + K
        ]


        canonical = (
            canonical_kmer(
                kmer
            )
        )


        individual_scores[
            i
        ] = (
            tenmer_score_dict.get(
                canonical,
                np.nan
            )
        )


    # ------------------------------------------------------------
    # Rolling sum.
    #
    # A valid cumulative score requires all 99 constituent
    # 10-mers to have a finite enrichment score.
    # ------------------------------------------------------------

    finite = np.isfinite(
        individual_scores
    )


    score_values = np.where(

        finite,

        individual_scores,

        0.0
    )


    cumulative_sum = np.concatenate(
        [
            [0.0],
            np.cumsum(
                score_values
            )
        ]
    )


    cumulative_valid = np.concatenate(
        [
            [0],
            np.cumsum(
                finite.astype(int)
            )
        ]
    )


    output = np.full(
        N_SCORE_POSITIONS,
        np.nan,
        dtype=float
    )


    for start in range(
        N_SCORE_POSITIONS
    ):

        end = (
            start
            +
            N_KMERS_PER_WINDOW
        )


        n_valid = (

            cumulative_valid[
                end
            ]

            -

            cumulative_valid[
                start
            ]
        )


        if (
            n_valid
            ==
            N_KMERS_PER_WINDOW
        ):

            output[
                start
            ] = (

                cumulative_sum[
                    end
                ]

                -

                cumulative_sum[
                    start
                ]
            )


    return output


# ================================================================
# 5. Score a collection of sequences
# ================================================================

def score_sequence_collection(
    sequences,
    print_every=5000
):

    output = np.empty(
        (
            len(sequences),
            N_SCORE_POSITIONS
        ),
        dtype=float
    )


    for i, sequence in enumerate(
        sequences
    ):

        output[
            i
        ] = (
            rolling_10mer_cumulative_score(
                sequence
            )
        )


        if (
            (i + 1)
            %
            print_every
            ==
            0
        ):

            print(
                "Scored:",
                i + 1,
                "/",
                len(sequences)
            )


    return output


# ================================================================
# 6. Position coordinates
# ================================================================
#
# A rolling score is assigned to the centre of its 108-bp
# window.
#
# For a window beginning at Python index j:
#
#     assigned position = j + 54
#
# This is the same convention used for the genome-wide 108-bp
# 10-mer cumulative-score analysis.
#
# The centre of the 297-bp sequence is index 148.
#
# Therefore:
#
#     relative position = j + 54 - 148
#
# for j = 0 ... 189.
#
# This gives:
#
#     -94 ... +95 bp
# ================================================================

score_position_bp = (

    np.arange(
        N_SCORE_POSITIONS
    )

    +
    54

    -
    148
)


assert (
    score_position_bp[0]
    ==
    -94
)


assert (
    score_position_bp[-1]
    ==
    95
)


# ================================================================
# 7. FIGURE 4.6a
# ================================================================
#
# IMPORTANT:
#
# Reuse the SAME reverse-translated sequences generated for
# Figure 4.5a.
#
# Do not generate another random set.
# ================================================================

print()
print(
    "Figure 4.6a: spatially varying codon usage"
)


tenmer_46a_spatial = (
    score_sequence_collection(
        random_peptide_spatial_sequences
    )
)


print()
print(
    "Figure 4.6a: overall codon usage"
)


tenmer_46a_uniform = (
    score_sequence_collection(
        random_peptide_uniform_sequences
    )
)


# ================================================================
# 8. Mean Figure 4.6a profiles
# ================================================================
#
# The thesis caption does not describe panel 4.6a as
# mean-centered, so retain the raw cumulative scores.
# ================================================================

mean_tenmer_46a_spatial = (
    np.nanmean(
        tenmer_46a_spatial,
        axis=0
    )
)


mean_tenmer_46a_uniform = (
    np.nanmean(
        tenmer_46a_uniform,
        axis=0
    )
)


figure_4_6a = pd.DataFrame(
    {
        "Position_bp":
            score_position_bp,

        "Spatially_varying":
            mean_tenmer_46a_spatial,

        "Uniform":
            mean_tenmer_46a_uniform
    }
)


# ================================================================
# 9. FIGURE 4.6b
# ================================================================
#
# Again reuse exactly the same sequence sets as Figure 4.5b.
# ================================================================

print()
print(
    "Figure 4.6b: native"
)


tenmer_46b_native = (
    score_sequence_collection(
        native_sequences_45b
    )
)


print()
print(
    "Figure 4.6b: random synonymous"
)


tenmer_46b_random = (
    score_sequence_collection(
        random_synonymous_sequences_45b
    )
)


print()
print(
    "Figure 4.6b: overall codon usage"
)


tenmer_46b_overall = (
    score_sequence_collection(
        overall_usage_sequences_45b
    )
)


print()
print(
    "Figure 4.6b: spatially varying codon usage"
)


tenmer_46b_spatial = (
    score_sequence_collection(
        spatial_usage_sequences_45b
    )
)


# ================================================================
# 10. Mean-center Figure 4.6b
# ================================================================
#
# The thesis caption explicitly describes Figure 4.6b as:
#
#     "Mean-centered cumulative nucleosome-SELEX
#      10-mer enrichment profiles"
#
# Mean-center EACH sequence's profile first, then average.
#
# This removes sequence-to-sequence differences in the absolute
# score and retains the positional pattern along the nucleosome.
# ================================================================

def mean_center_profiles(
    matrix
):

    row_means = np.nanmean(
        matrix,
        axis=1,
        keepdims=True
    )


    return (
        matrix
        -
        row_means
    )


tenmer_46b_native_centered = (
    mean_center_profiles(
        tenmer_46b_native
    )
)


tenmer_46b_random_centered = (
    mean_center_profiles(
        tenmer_46b_random
    )
)


tenmer_46b_overall_centered = (
    mean_center_profiles(
        tenmer_46b_overall
    )
)


tenmer_46b_spatial_centered = (
    mean_center_profiles(
        tenmer_46b_spatial
    )
)


# ================================================================
# 11. Mean Figure 4.6b profiles
# ================================================================

mean_tenmer_46b_native = (
    np.nanmean(
        tenmer_46b_native_centered,
        axis=0
    )
)


mean_tenmer_46b_random = (
    np.nanmean(
        tenmer_46b_random_centered,
        axis=0
    )
)


mean_tenmer_46b_overall = (
    np.nanmean(
        tenmer_46b_overall_centered,
        axis=0
    )
)


mean_tenmer_46b_spatial = (
    np.nanmean(
        tenmer_46b_spatial_centered,
        axis=0
    )
)


figure_4_6b = pd.DataFrame(
    {
        "Position_bp":
            score_position_bp,

        "Native":
            mean_tenmer_46b_native,

        "Random_synonymous":
            mean_tenmer_46b_random,

        "Overall_codon_usage":
            mean_tenmer_46b_overall,

        "Spatially_varying":
            mean_tenmer_46b_spatial
    }
)


# ================================================================
# 12. Useful raw means
# ================================================================
#
# Retain the uncentered means too. They are not the stated
# Figure 4.6b quantity, but are useful as a reconstruction check.
# ================================================================

figure_4_6b_raw = pd.DataFrame(
    {
        "Position_bp":
            score_position_bp,

        "Native":
            np.nanmean(
                tenmer_46b_native,
                axis=0
            ),

        "Random_synonymous":
            np.nanmean(
                tenmer_46b_random,
                axis=0
            ),

        "Overall_codon_usage":
            np.nanmean(
                tenmer_46b_overall,
                axis=0
            ),

        "Spatially_varying":
            np.nanmean(
                tenmer_46b_spatial,
                axis=0
            )
    }
)


# ================================================================
# 13. Final checks
# ================================================================

assert (
    tenmer_46a_spatial.shape[1]
    ==
    190
)


assert (
    tenmer_46a_uniform.shape[1]
    ==
    190
)


assert (
    tenmer_46b_native.shape[1]
    ==
    190
)


assert len(
    figure_4_6a
) == 190


assert len(
    figure_4_6b
) == 190


# The mean of each mean-centered profile should be approximately
# zero, apart from missing values / floating-point error.

native_center_check = (
    np.nanmean(
        tenmer_46b_native_centered,
        axis=1
    )
)


print()
print(
    "Figure 4.6 calculations complete."
)


print()
print(
    "Figure 4.6a sequences:",
    len(
        random_peptide_spatial_sequences
    )
)


print(
    "Figure 4.6b native sequences:",
    len(
        native_sequences_45b
    )
)


print(
    "Rolling positions:",
    N_SCORE_POSITIONS
)


print(
    "Relative coordinate range:",
    score_position_bp[0],
    "to",
    score_position_bp[-1],
    "bp"
)


print()
print(
    "Largest absolute residual mean after "
    "4.6b native mean-centering:",
    np.nanmax(
        np.abs(
            native_center_check
        )
    )
)


print()
print(
    "Figure 4.6a:"
)


print(
    figure_4_6a.head()
)


print()
print(
    "Figure 4.6b:"
)


print(
    figure_4_6b.head()
)