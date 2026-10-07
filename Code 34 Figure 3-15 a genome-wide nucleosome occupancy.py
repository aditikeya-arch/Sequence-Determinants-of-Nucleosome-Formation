

"""
SCRIPT NAME:
Code 34 Figure 3-15 genome-wide nucleosome occupancy.py

WHAT THIS SCRIPT DOES:
Calculates genome-wide nucleosome occupancy predicted from the
108-bp 10-mer cumulative score and a hard-rod steric exclusion model.

For every position in the 16 Saccharomyces cerevisiae chromosomes:

1. Calculate the 108-bp 10-mer cumulative score.
2. Convert the position-based score into a score for each possible
   147-bp nucleosome start.
3. Standardise scores genome-wide.
4. Convert scores into statistical weights.
5. Apply the exact one-dimensional hard-rod model.
6. Convert nucleosome start probabilities into per-base occupancy.
7. Compare predicted occupancy with measured in-vitro occupancy.

Figure 3.15a uses held-out chromosomes 13-16.

No plotting is performed here. The script produces the arrays needed
to make Figure 3.15a and variables that can also be used for
Figure 3.15b.


THESIS SECTION / FIGURE:
Figure 3.15


INPUT FILES:
GSM351491_InVitro_normalized.tab

Expected columns in this file:
    chromosome number
    genomic position
    measured occupancy

The genomic positions in this file are treated as 1-based.


VARIABLES THAT MUST ALREADY EXIST:
sequences_data

    List of 16 strings containing the complete sequences of
    S. cerevisiae chromosomes 1-16, in chromosome order.

df_enrichment_rc

    Reverse-complement-collapsed 10-mer enrichment table generated
    in:

    Code 20 Figure 2-25a 10-mer enrichment distribution.py

    It must contain:
        kmer
        log2_enrichment


FUNCTIONS THAT MUST ALREADY EXIST:
canonical_kmer

    From:
    Code 20 Figure 2-25a 10-mer enrichment distribution.py


OTHER SCRIPTS THAT MUST BE RUN FIRST:
Code 20 Figure 2-25a 10-mer enrichment distribution.py


OUTPUT VARIABLES:
yeast_10mer_CS
    Dictionary containing the 108-bp 10-mer cumulative score at
    genomic positions in each chromosome.

zhang_occ
    Dictionary containing measured in-vitro occupancy.

score_mean
score_sd
    Genome-wide mean and standard deviation of possible nucleosome
    start scores.

yeast_pred_occ
    Predicted per-base nucleosome occupancy for all 16 chromosomes.

figure_3_15a
    Data required for Figure 3.15a on held-out chromosomes 13-16.

figure_3_15a_r
    Pearson correlation for Figure 3.15a.


NOTES / THINGS I DON'T REMEMBER:
The historic analysis uses alpha = 0.1 and mu = -6.5 for the final
comparison with held-out chromosomes.

The historic file does not contain the code that fitted these two
parameters. The thesis states that chromosomes 1-12 were used for
training and chromosomes 13-16 were held out for testing.

Therefore the values alpha = 0.1 and mu = -6.5 are preserved here
rather than refitted.

The historic code refers to the genomic 10-mer cumulative score as
"NFP". Here it is called yeast_10mer_CS to distinguish it from the
experimentally measured nucleosome formation propensity used earlier
in Chapter 3.
"""


import numpy as np
import pandas as pd

from scipy.stats import pearsonr

try:
    from numba import njit
except ImportError:
    raise ImportError(
        "This script requires numba. "
        "For example: conda install numba"
    )


# ================================================================
# 1. Constants
# ================================================================

K = 10

# Going forward, predicted nucleosome formation propensity from the
# 10-mer model is calculated using the centred 108-bp window.
SCORE_WINDOW = 108

# Number of overlapping 10-mers in a 108-bp region:
#
#     108 - 10 + 1 = 99
#
N_KMERS_PER_WINDOW = SCORE_WINDOW - K + 1

# The historic genomic calculation assigns the score of
# seq[j:j+108] to genomic position j + 54.
ASSIGNED_POSITION = 54

ROD_LENGTH = 147

ZHANG_FILE = "GSM351491_InVitro_normalized.tab"


# Final parameters used in the historic Figure 3.15 analysis.
ALPHA = 0.1
MU = -6.5


# Training/test chromosome split described in the thesis.
TRAIN_CHROMS = [f"chr{i}" for i in range(1, 13)]
TEST_CHROMS = [f"chr{i}" for i in range(13, 17)]


# ================================================================
# 2. Check prerequisites
# ================================================================

assert len(sequences_data) == 16

for chromosome_sequence in sequences_data:
    assert isinstance(chromosome_sequence, str)


required_columns = {
    "kmer",
    "log2_enrichment"
}

missing_columns = required_columns - set(df_enrichment_rc.columns)

if missing_columns:
    raise ValueError(
        "df_enrichment_rc is missing columns: "
        + ", ".join(sorted(missing_columns))
    )


# ================================================================
# 3. Make the 10-mer score dictionary
# ================================================================
#
# df_enrichment_rc contains one representative of each
# reverse-complement pair.
#
# canonical_kmer() was defined in Code 20.
#

kmer_score = dict(
    zip(
        df_enrichment_rc["kmer"],
        df_enrichment_rc["log2_enrichment"]
    )
)


# ================================================================
# 4. Calculate the 108-bp 10-mer cumulative score genome-wide
# ================================================================

def compute_yeast_10mer_CS(
    sequences_data,
    kmer_score,
    k=10,
    window=108,
    assigned_position=54
):
    """
    Calculate the 108-bp 10-mer cumulative score at every genomic
    position for which the complete scoring window is available.

    For a window:

        sequence[j : j + 108]

    all 99 overlapping 10-mers are scored:

        j:j+10
        j+1:j+11
        ...
        j+98:j+108

    The sum is assigned to genomic position:

        j + 54

    Missing 10-mer enrichment values cause the corresponding
    108-bp score to be NaN.

    Returns
    -------
    dictionary

        chr1 -> numpy array
        ...
        chr16 -> numpy array

    Each array has the same length as its chromosome.
    """

    n_kmers_per_window = window - k + 1

    yeast_scores = {}

    for chromosome_index, chromosome_sequence in enumerate(
        sequences_data
    ):

        chrom = f"chr{chromosome_index + 1}"

        sequence = str(chromosome_sequence).upper()

        chromosome_length = len(sequence)

        chromosome_scores = np.full(
            chromosome_length,
            np.nan,
            dtype=float
        )


        # --------------------------------------------------------
        # Score every individual genomic 10-mer
        # --------------------------------------------------------

        n_kmer_starts = chromosome_length - k + 1

        individual_kmer_scores = np.full(
            n_kmer_starts,
            np.nan,
            dtype=float
        )

        for start in range(n_kmer_starts):

            tenmer = sequence[start:start + k]

            if "N" in tenmer:
                continue

            canonical = canonical_kmer(tenmer)

            individual_kmer_scores[start] = kmer_score.get(
                canonical,
                np.nan
            )


        # --------------------------------------------------------
        # Efficient rolling sum of 99 consecutive 10-mer scores
        # --------------------------------------------------------
        #
        # A valid 108-bp cumulative score requires all 99 10-mers
        # to have valid enrichment scores.
        #

        valid = ~np.isnan(individual_kmer_scores)

        filled_scores = np.where(
            valid,
            individual_kmer_scores,
            0.0
        )


        cumulative_score = np.concatenate(
            [
                [0.0],
                np.cumsum(filled_scores)
            ]
        )

        cumulative_valid = np.concatenate(
            [
                [0],
                np.cumsum(valid.astype(int))
            ]
        )


        rolling_sum = (
            cumulative_score[n_kmers_per_window:]
            -
            cumulative_score[:-n_kmers_per_window]
        )


        rolling_valid_count = (
            cumulative_valid[n_kmers_per_window:]
            -
            cumulative_valid[:-n_kmers_per_window]
        )


        # Require all 99 10-mers.
        rolling_sum[
            rolling_valid_count != n_kmers_per_window
        ] = np.nan


        # rolling_sum[j] corresponds to:
        #
        #     sequence[j:j+108]
        #
        # Historic convention:
        #
        #     assign this score to j + 54
        #

        positions = (
            np.arange(len(rolling_sum))
            +
            assigned_position
        )

        chromosome_scores[positions] = rolling_sum

        yeast_scores[chrom] = chromosome_scores

        print(
            f"{chrom}: calculated 108-bp 10-mer cumulative score "
            f"({chromosome_length:,} bp)"
        )


    return yeast_scores


yeast_10mer_CS = compute_yeast_10mer_CS(
    sequences_data=sequences_data,
    kmer_score=kmer_score,
    k=K,
    window=SCORE_WINDOW,
    assigned_position=ASSIGNED_POSITION
)


# ================================================================
# 5. Load measured in-vitro nucleosome occupancy
# ================================================================
#
# Historic input:
#
#     GSM351491_InVitro_normalized.tab
#
# Each row contains:
#
#     chromosome_number    position    occupancy
#
# Positions are converted from 1-based to 0-based coordinates.
#

zhang_occ = {
    f"chr{i + 1}": np.zeros(
        len(sequence),
        dtype=float
    )
    for i, sequence in enumerate(sequences_data)
}


with open(ZHANG_FILE) as file:

    # Skip header.
    next(file)

    for line in file:

        if not line.strip():
            continue

        chrom_num, position, value = line.strip().split()

        chrom = f"chr{int(chrom_num)}"

        # File uses 1-based coordinates.
        position = int(position) - 1

        value = float(value)

        if chrom not in zhang_occ:
            continue

        if 0 <= position < len(zhang_occ[chrom]):
            zhang_occ[chrom][position] = value


# ================================================================
# 6. Hard-rod model
# ================================================================

@njit
def logaddexp_numba(a, b):
    """
    Numerically stable log(exp(a) + exp(b)).
    """

    if a == -np.inf:
        return b

    if b == -np.inf:
        return a

    maximum = a if a > b else b

    return maximum + np.log(
        np.exp(a - maximum)
        +
        np.exp(b - maximum)
    )


@njit
def hard_rod_start_probs(
    logw,
    rod_len=147
):
    """
    Exact one-dimensional hard-rod partition function.

    Parameters
    ----------
    logw

        logw[i] is the log statistical weight of placing a
        nucleosome beginning at genomic start position i.

    rod_len

        Nucleosome footprint in bp.

    Returns
    -------
    p_start

        Probability of a nucleosome beginning at every possible
        genomic start position.
    """

    n_starts = len(logw)

    chromosome_length = (
        n_starts
        +
        rod_len
        -
        1
    )


    # ------------------------------------------------------------
    # Forward and backward partition functions
    # ------------------------------------------------------------

    logF = np.empty(
        chromosome_length + 1
    )

    logB = np.empty(
        chromosome_length + rod_len + 1
    )


    for i in range(chromosome_length + 1):
        logF[i] = -np.inf

    for i in range(chromosome_length + rod_len + 1):
        logB[i] = 0.0


    logF[0] = 0.0


    # ------------------------------------------------------------
    # Forward recursion
    # ------------------------------------------------------------

    for t in range(1, chromosome_length + 1):

        # No nucleosome ending here.
        value = logF[t - 1]

        # A nucleosome whose 147-bp footprint ends here.
        start = t - rod_len

        if 0 <= start < n_starts:

            if logw[start] != -np.inf:

                value_with_nucleosome = (
                    logw[start]
                    +
                    logF[t - rod_len]
                )

                value = logaddexp_numba(
                    value,
                    value_with_nucleosome
                )

        logF[t] = value


    # ------------------------------------------------------------
    # Backward recursion
    # ------------------------------------------------------------

    for t in range(
        chromosome_length - 1,
        -1,
        -1
    ):

        value = logB[t + 1]

        if t < n_starts:

            if logw[t] != -np.inf:

                value_with_nucleosome = (
                    logw[t]
                    +
                    logB[t + rod_len]
                )

                value = logaddexp_numba(
                    value,
                    value_with_nucleosome
                )

        logB[t] = value


    logZ = logF[chromosome_length]


    # ------------------------------------------------------------
    # Probability of a nucleosome starting at each position
    # ------------------------------------------------------------

    p_start = np.empty(n_starts)


    for i in range(n_starts):

        if logw[i] == -np.inf:

            p_start[i] = 0.0

        else:

            p_start[i] = np.exp(
                logw[i]
                +
                logF[i]
                +
                logB[i + rod_len]
                -
                logZ
            )


    return p_start


# ================================================================
# 7. Convert centre-based 10-mer scores to nucleosome-start scores
# ================================================================

def start_scores_from_center_scores(
    center_scores,
    rod_len=147
):
    """
    Assign a 10-mer cumulative score to every possible 147-bp
    nucleosome start.

    Historic convention:

        nucleosome centre = start + 147 // 2
                           = start + 73
    """

    chromosome_length = len(center_scores)

    n_starts = (
        chromosome_length
        -
        rod_len
        +
        1
    )

    starts = np.arange(n_starts)

    centers = (
        starts
        +
        rod_len // 2
    )

    return center_scores[centers]


# ================================================================
# 8. Convert start probabilities into per-base occupancy
# ================================================================

def start_probs_to_occupancy(
    p_start,
    chromosome_length,
    rod_len=147
):
    """
    Per-base occupancy is the sum of the probabilities of all
    nucleosomes whose 147-bp footprints cover that base.
    """

    occupancy = np.convolve(
        p_start,
        np.ones(rod_len),
        mode="full"
    )[:chromosome_length]

    return occupancy


# ================================================================
# 9. Genome-wide score standardisation
# ================================================================
#
# The historic implementation standardises the possible nucleosome
# start scores using one mean and standard deviation calculated
# across all 16 chromosomes.
#

all_start_scores = []


for chromosome_number in range(1, 17):

    chrom = f"chr{chromosome_number}"

    scores = start_scores_from_center_scores(
        yeast_10mer_CS[chrom],
        rod_len=ROD_LENGTH
    )

    all_start_scores.append(
        scores[~np.isnan(scores)]
    )


all_start_scores = np.concatenate(
    all_start_scores
)


score_mean = np.mean(
    all_start_scores
)

score_sd = np.std(
    all_start_scores
)


print(
    "Genome-wide nucleosome-start score mean:",
    score_mean
)

print(
    "Genome-wide nucleosome-start score SD:",
    score_sd
)


# ================================================================
# 10. Predict occupancy for one chromosome
# ================================================================

def compute_predicted_occupancy_for_chrom(
    center_scores,
    alpha,
    mu,
    score_mean,
    score_sd,
    rod_len=147
):
    """
    Predict per-base nucleosome occupancy from the 10-mer cumulative
    score using the hard-rod steric-exclusion model.
    """

    # One score for every possible nucleosome start.
    score_start = start_scores_from_center_scores(
        center_scores,
        rod_len=rod_len
    )


    valid = ~np.isnan(
        score_start
    )


    # ------------------------------------------------------------
    # Standardise the 10-mer cumulative score
    # ------------------------------------------------------------

    z = np.full_like(
        score_start,
        np.nan,
        dtype=float
    )

    z[valid] = (
        score_start[valid]
        -
        score_mean
    ) / score_sd


    # ------------------------------------------------------------
    # Statistical weight
    #
    # Historic implementation:
    #
    #     log(w_i) = alpha * z_i + mu
    #
    # Thus larger 10-mer cumulative score gives larger intrinsic
    # nucleosome weight.
    # ------------------------------------------------------------

    logw = np.full(
        len(score_start),
        -np.inf,
        dtype=float
    )

    logw[valid] = (
        alpha * z[valid]
        +
        mu
    )


    # ------------------------------------------------------------
    # Enforce steric exclusion
    # ------------------------------------------------------------

    p_start = hard_rod_start_probs(
        logw,
        rod_len=rod_len
    )


    # ------------------------------------------------------------
    # Convert start probabilities to per-base occupancy
    # ------------------------------------------------------------

    predicted_occupancy = start_probs_to_occupancy(
        p_start,
        chromosome_length=len(center_scores),
        rod_len=rod_len
    )


    return predicted_occupancy


# ================================================================
# 11. Generate predicted occupancy for all chromosomes
# ================================================================

yeast_pred_occ = {}


for chromosome_number in range(1, 17):

    chrom = f"chr{chromosome_number}"

    yeast_pred_occ[chrom] = (
        compute_predicted_occupancy_for_chrom(
            center_scores=yeast_10mer_CS[chrom],
            alpha=ALPHA,
            mu=MU,
            score_mean=score_mean,
            score_sd=score_sd,
            rod_len=ROD_LENGTH
        )
    )

    print(
        f"{chrom}: predicted occupancy calculated"
    )


# ================================================================
# 12. Figure 3.15a
#     held-out chromosomes 13-16 only
# ================================================================
#
# The model parameters were selected using chromosomes 1-12.
#
# Figure 3.15a evaluates the resulting model on chromosomes 13-16.
#

predicted_values = []
measured_values = []
chromosome_labels = []
genomic_positions = []


for chrom in TEST_CHROMS:

    predicted = yeast_pred_occ[chrom]

    measured = zhang_occ[chrom]


    mask = (
        ~np.isnan(predicted)
        &
        ~np.isnan(measured)
    )


    predicted_values.append(
        predicted[mask]
    )

    measured_values.append(
        measured[mask]
    )


    chromosome_labels.extend(
        [chrom] * np.sum(mask)
    )


    genomic_positions.extend(
        np.where(mask)[0]
    )


predicted_values = np.concatenate(
    predicted_values
)

measured_values = np.concatenate(
    measured_values
)


# ================================================================
# 13. Correlation for Figure 3.15a
# ================================================================

figure_3_15a_r, figure_3_15a_p = pearsonr(
    predicted_values,
    measured_values
)


print()
print("Figure 3.15a")
print("------------------------------")

print(
    "Test chromosomes:",
    ", ".join(TEST_CHROMS)
)

print(
    "Number of genomic positions:",
    len(predicted_values)
)

print(
    f"Pearson r = {figure_3_15a_r:.4f}"
)

print(
    f"p = {figure_3_15a_p:.4e}"
)


# ================================================================
# 14. Data for Figure 3.15a
# ================================================================
#
# This dataframe contains the raw x/y data required for the density
# scatter plot.
#

figure_3_15a = pd.DataFrame(
    {
        "chromosome": chromosome_labels,
        "position_0based": genomic_positions,
        "predicted_occupancy": predicted_values,
        "measured_occupancy": measured_values
    }
)


# ================================================================
# 15. Optional version excluding measured zero values
# ================================================================
#
# The historic code also examined the relationship after removing
# the exact zero band in the measured occupancy data.
#
# This is retained as an auxiliary analysis rather than replacing
# the main Figure 3.15a dataset.
#

nonzero_mask = (
    figure_3_15a["measured_occupancy"]
    !=
    0
)


figure_3_15a_nonzero = (
    figure_3_15a.loc[nonzero_mask]
    .reset_index(drop=True)
)


figure_3_15a_nonzero_r, figure_3_15a_nonzero_p = pearsonr(
    figure_3_15a_nonzero["predicted_occupancy"],
    figure_3_15a_nonzero["measured_occupancy"]
)


print()
print(
    "Figure 3.15a, excluding exact measured zeros:"
)

print(
    f"Pearson r = {figure_3_15a_nonzero_r:.4f}"
)