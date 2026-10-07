
"""
Code 20 Figure 2-25a 10-mer enrichment distribution.py
=======================================================

WHAT THIS SCRIPT DOES
---------------------
Constructs the reverse-complement-collapsed 10-mer
enrichment table used for Figure 2.25 and subsequent
10-mer analyses.

The calculation compares the frequency of each 10-mer
class in Round 6 with its frequency in Round 1.

All reads are represented through Counts Per Million
weighting of the unique sequences in the input files.


ANALYSIS
--------
For each unique sequence in R1 and R6:

    1. Convert the sequence to uppercase.

    2. Ignore sequences containing N.

    3. Remove the 17-bp constant region from each end,
       retaining the central 113-bp variable region.

    4. Extract all 104 overlapping 10-mers.

    5. Add the Counts Per Million value of the parent
       sequence to the count of every 10-mer occurrence.

The resulting oriented 10-mer counts are then collapsed
with their reverse complements.

For each reverse-complement pair:

    collapsed count =
        count(kmer) + count(reverse complement)

A single canonical sequence is used to represent each
reverse-complement pair.

The collapsed counts are converted to frequencies
separately for R1 and R6.

Enrichment is then calculated as:

    enrichment =
        frequency_R6 / frequency_R1

and:

    log2_enrichment =
        log2(enrichment)


REVERSE-COMPLEMENT COLLAPSING
-----------------------------
For each 10-mer, its reverse complement is calculated.

The lexicographically smaller of:

    kmer
    reverse_complement(kmer)

is used as the canonical representative.

Because 10 is even, some 10-mers are their own reverse
complements. These self-reverse-complementary sequences
are counted only once.

Across the complete 10-mer sequence space:

    4^10 = 1,048,576 oriented 10-mers

which collapse into:

    524,800 reverse-complement classes.


IMPORTANT
---------
Reverse-complement collapsing is performed on the
CPM-weighted counts BEFORE frequencies and enrichment
are calculated.

Enrichment values for reverse-complement partners are
therefore not averaged.

No pseudocount is added.

A reverse-complement class must have non-zero frequency
in both R1 and R6 to enter the final enrichment table.


INPUT FILES
-----------
R1.csv
R6.csv

Each file must contain:

    Sequence
    Counts Per Million


OUTPUTS
-------
counts_r1
counts_r6

    CPM-weighted oriented 10-mer counts.


counts_r1_rc
counts_r6_rc

    Reverse-complement-collapsed CPM-weighted counts.


df_enrichment_rc

    Reverse-complement-collapsed enrichment table
    containing:

        kmer
        freq_R1
        freq_R6
        enrichment
        log2_enrichment


figure_2_25a

    Data required for the Figure 2.25a distribution.


top_0_1_percent_10mers

    Highest-ranked 0.1% of reverse-complement-collapsed
    10-mers.


No plotting is performed.
"""


import numpy as np
import pandas as pd

from collections import defaultdict


# ============================================================
# 1. Parameters
# ============================================================

K = 10

TRIM_LEFT = 17

TRIM_RIGHT = 17

VARIABLE_LENGTH = 113


R1_FILE = "R1.csv"

R6_FILE = "R6.csv"


# ============================================================
# 2. Load Round 1 and Round 6
# ============================================================

print(
    "Loading data..."
)


df_r1 = pd.read_csv(

    R1_FILE,

    usecols=[
        "Sequence",
        "Counts Per Million",
    ]
)


df_r6 = pd.read_csv(

    R6_FILE,

    usecols=[
        "Sequence",
        "Counts Per Million",
    ]
)


# ============================================================
# 3. Count CPM-weighted oriented 10-mers
# ============================================================

def get_kmer_counts(
    dataframe
):

    """
    Count all overlapping oriented 10-mers in the
    central 113-bp region.

    Each occurrence contributes the Counts Per Million
    value of its parent sequence.
    """

    counts = defaultdict(
        float
    )


    for sequence, weight in zip(

        dataframe[
            "Sequence"
        ],

        dataframe[
            "Counts Per Million"
        ]
    ):

        sequence = str(
            sequence
        ).upper()


        # ----------------------------------------------------
        # Ignore sequences containing ambiguous bases
        # ----------------------------------------------------

        if "N" in sequence:

            continue


        # ----------------------------------------------------
        # Retain central 113-bp variable region
        # ----------------------------------------------------

        sequence = sequence[

            TRIM_LEFT:

            len(sequence)
            -
            TRIM_RIGHT
        ]


        # Only correctly sized sequences contribute.

        if len(
            sequence
        ) != VARIABLE_LENGTH:

            continue


        # ----------------------------------------------------
        # Extract all overlapping 10-mers
        #
        # 113 - 10 + 1 = 104
        # ----------------------------------------------------

        for i in range(

            len(sequence)
            -
            K
            +
            1
        ):

            kmer = sequence[
                i:
                i + K
            ]


            counts[
                kmer
            ] += float(
                weight
            )


    return counts


# ============================================================
# 4. Count R1 and R6 oriented 10-mers
# ============================================================

print(
    "Counting Round 1 10-mers..."
)


counts_r1 = get_kmer_counts(
    df_r1
)


print(
    "Counting Round 6 10-mers..."
)


counts_r6 = get_kmer_counts(
    df_r6
)


# ============================================================
# 5. Reverse-complement functions
# ============================================================

COMPLEMENT = str.maketrans(

    "ACGT",

    "TGCA"
)


def reverse_complement(
    sequence
):

    """
    Return the reverse complement of a DNA sequence.
    """

    return (

        sequence

        .translate(
            COMPLEMENT
        )

        [::-1]
    )


def canonical_kmer(
    kmer
):

    """
    Return one canonical representative for a 10-mer
    and its reverse complement.

    The lexicographically smaller sequence is used.
    """

    rc = reverse_complement(
        kmer
    )


    return min(
        kmer,
        rc
    )


# ============================================================
# 6. Collapse oriented counts by reverse complement
# ============================================================

def collapse_reverse_complements(
    counts
):

    """
    Collapse CPM-weighted oriented k-mer counts into
    reverse-complement classes.

    Each oriented k-mer is mapped to its canonical
    representative.

    This automatically handles self-reverse-complementary
    10-mers correctly because each oriented sequence is
    visited only once.
    """

    collapsed = defaultdict(
        float
    )


    for kmer, count in counts.items():

        canonical = canonical_kmer(
            kmer
        )


        collapsed[
            canonical
        ] += count


    return collapsed


print(
    "Collapsing reverse complements..."
)


counts_r1_rc = collapse_reverse_complements(
    counts_r1
)


counts_r6_rc = collapse_reverse_complements(
    counts_r6
)


# ============================================================
# 7. Convert collapsed counts to frequencies
# ============================================================

total_r1_rc = sum(
    counts_r1_rc.values()
)


total_r6_rc = sum(
    counts_r6_rc.values()
)


freq_r1_rc = {

    kmer:
        count
        /
        total_r1_rc

    for kmer, count
    in counts_r1_rc.items()
}


freq_r6_rc = {

    kmer:
        count
        /
        total_r6_rc

    for kmer, count
    in counts_r6_rc.items()
}


# ============================================================
# 8. Calculate R6 / R1 enrichment
# ============================================================

rows = []


# Iterate through reverse-complement classes observed
# in Round 6.

for kmer in freq_r6_rc:


    # Only retain classes also observed in Round 1.

    if (

        kmer in freq_r1_rc

        and

        freq_r1_rc[
            kmer
        ] > 0

        and

        freq_r6_rc[
            kmer
        ] > 0
    ):

        enrichment = (

            freq_r6_rc[
                kmer
            ]

            /

            freq_r1_rc[
                kmer
            ]
        )


        rows.append({

            "kmer":
                kmer,

            "freq_R1":
                freq_r1_rc[
                    kmer
                ],

            "freq_R6":
                freq_r6_rc[
                    kmer
                ],

            "enrichment":
                enrichment,

            "log2_enrichment":
                np.log2(
                    enrichment
                ),
        })


# ============================================================
# 9. Complete reverse-complement-collapsed enrichment table
# ============================================================

df_enrichment_rc = pd.DataFrame(
    rows
)


df_enrichment_rc = (

    df_enrichment_rc

    .sort_values(
        "enrichment",
        ascending=False
    )

    .reset_index(
        drop=True
    )
)


# ============================================================
# 10. Figure 2.25a data
# ============================================================

figure_2_25a = (

    df_enrichment_rc[
        [
            "kmer",
            "log2_enrichment",
        ]
    ]

    .copy()
)


# ============================================================
# 11. Top 0.1%
# ============================================================

n_top_0_1_percent = max(

    1,

    int(

        0.001

        *

        len(
            df_enrichment_rc
        )
    )
)


top_0_1_percent_10mers = (

    df_enrichment_rc

    .head(
        n_top_0_1_percent
    )

    .copy()
)


# ============================================================
# 12. Additional ranked groups used in Figure 2.25
# ============================================================

# With 524,800 reverse-complement classes:
#
#     0.1% = 524.8
#
# The Figure 2.25 ranked groups contain approximately
# 525 10-mers each.


n_0_1_percent = max(

    1,

    int(

        0.001

        *

        len(
            df_enrichment_rc
        )
    )
)


top_0_to_0_1_percent = (

    df_enrichment_rc

    .iloc[
        0:
        n_0_1_percent
    ]

    .copy()
)


top_0_1_to_0_2_percent = (

    df_enrichment_rc

    .iloc[
        n_0_1_percent:
        2 * n_0_1_percent
    ]

    .copy()
)


top_0_2_to_0_3_percent = (

    df_enrichment_rc

    .iloc[
        2 * n_0_1_percent:
        3 * n_0_1_percent
    ]

    .copy()
)


bottom_0_1_percent = (

    df_enrichment_rc

    .tail(
        n_0_1_percent
    )

    .copy()
)


# ============================================================
# 13. Useful checks
# ============================================================

print(
    "\nNumber of oriented 10-mers observed in R1:",
    len(
        counts_r1
    )
)


print(
    "Number of oriented 10-mers observed in R6:",
    len(
        counts_r6
    )
)


print(
    "\nNumber of reverse-complement classes "
    "observed in R1:",
    len(
        counts_r1_rc
    )
)


print(
    "Number of reverse-complement classes "
    "observed in R6:",
    len(
        counts_r6_rc
    )
)


print(
    "\nNumber of reverse-complement classes "
    "in enrichment table:",
    len(
        df_enrichment_rc
    )
)


print(
    "Expected complete reverse-complement sequence space:",
    524800
)


print(
    "\nTop 0.1%:",
    n_top_0_1_percent,
    "10-mers"
)


# ============================================================
# 14. Check complete reverse-complement sequence space
# ============================================================

# If every possible 10-mer was observed in both rounds,
# the final table should contain exactly 524,800
# reverse-complement classes.

if len(
    df_enrichment_rc
) == 524800:

    print(
        "All 524,800 reverse-complement classes are present."
    )

else:

    print(

        "WARNING: enrichment table contains",

        len(
            df_enrichment_rc
        ),

        "classes rather than 524,800."
    )


# ============================================================
# 15. Highest enrichment values
# ============================================================

print(
    "\nHighest enrichment values:"
)


print(

    df_enrichment_rc[
        [
            "kmer",
            "log2_enrichment",
        ]
    ]

    .head(
        30
    )

    .to_string(
        index=False
    )
)


# ============================================================
# 16. Lowest enrichment values
# ============================================================

print(
    "\nLowest enrichment values:"
)


print(

    df_enrichment_rc[
        [
            "kmer",
            "log2_enrichment",
        ]
    ]

    .tail(
        30
    )

    .to_string(
        index=False
    )
)


# ============================================================
# OUTPUT SUMMARY
# ============================================================

# CPM-weighted oriented counts:
#
#     counts_r1
#     counts_r6
#
#
# Reverse-complement-collapsed counts:
#
#     counts_r1_rc
#     counts_r6_rc
#
#
# Reverse-complement-collapsed enrichment table:
#
#     df_enrichment_rc
#
# Columns:
#
#     kmer
#     freq_R1
#     freq_R6
#     enrichment
#     log2_enrichment
#
#
# Figure 2.25a:
#
#     figure_2_25a
#
#
# Ranked groups:
#
#     top_0_1_percent_10mers
#
#     top_0_to_0_1_percent
#
#     top_0_1_to_0_2_percent
#
#     top_0_2_to_0_3_percent
#
#     bottom_0_1_percent
#
#
# IMPORTANT:
#
#     - all reads are represented through CPM weighting
#
#     - only the central 113 bp is analysed
#
#     - 104 overlapping 10-mers occur per valid sequence
#
#     - oriented 10-mers are counted first
#
#     - reverse complements are then collapsed by summing
#       their CPM-weighted counts
#
#     - frequencies are calculated AFTER reverse-complement
#       collapsing
#
#     - enrichment is calculated AFTER reverse-complement
#       collapsing
#
#     - no pseudocount is used
#
#     - a reverse-complement class must occur in both
#       R1 and R6 to enter the enrichment table
#
#
# The main downstream table is:
#
#     df_enrichment_rc
#
# This is the input expected by:
#
#     Code 21 Figure 2-26 10-mer GC and poly dAdT analysis.py
#
#
# No plotting is performed.