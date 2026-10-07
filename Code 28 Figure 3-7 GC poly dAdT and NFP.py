

"""
Code 28 Figure 3-7 GC poly dAdT and NFP.py
===========================================

WHAT THIS SCRIPT DOES
---------------------
Generates the data underlying both panels of Figure 3.7.

LEFT PANEL
----------
Compares GC fraction in the central 113 bp of each
sequence with experimentally measured nucleosome
formation propensity (NFP).

RIGHT PANEL
-----------
Compares poly(dA:dT) fraction in the central 113 bp of
each sequence with experimentally measured NFP.

For both panels, each point represents one sequence from
the 2,000-member competitive nucleosome reconstitution
library.

The 200 SELEX sequences and Widom 601 are identified
separately so that they can be highlighted when plotting.

Pearson correlations are calculated for both relationships.


VARIABLES THAT MUST ALREADY EXIST
---------------------------------
seq

    Ordered list containing the 2,000 experimental
    sequences.

    Every sequence must be 147 bp long.


NFP

    Array containing the experimentally measured
    nucleosome formation propensity of the same 2,000
    sequences, in the same order.

    seq and NFP are generated in:

    Code 26 Figure 3-5 nucleosome formation propensity
    by sequence group.py


FUNCTIONS THAT MUST ALREADY EXIST
---------------------------------
None.


OTHER SCRIPTS THAT MUST BE RUN FIRST
------------------------------------
Code 26 Figure 3-5 nucleosome formation propensity
by sequence group.py

Alternatively, seq and NFP can be supplied directly.


SEQUENCE REGION ANALYSED
------------------------
All sequence-composition measurements are calculated
over the central 113 bp:

    sequence[17:130]

The 17-bp adapters at either end of the 147-bp
experimental construct are therefore excluded.


GC DEFINITION
-------------
GC fraction is:

    number of G or C bases / 113


POLY(dA:dT) DEFINITION
----------------------
A poly(dA:dT) tract is defined as an uninterrupted
homopolymeric run of A or T of length >= 5 bp.

Examples:

    AAAAA       qualifies
    TTTTTT      qualifies
    AAAAT       does not qualify
    ATATAT      does not qualify

The poly(dA:dT) fraction is the fraction of the central
113 bases that participate in qualifying A or T runs.


OUTPUTS
-------
figure_3_7

    Complete sequence-level data containing:

        Library_position
        Sequence
        Central_113
        GC_fraction
        Poly_dAdT_bases
        Poly_dAdT_fraction
        NFP
        Plot_class


figure_3_7a

    Data underlying the GC versus NFP panel.


figure_3_7b

    Data underlying the poly(dA:dT) versus NFP panel.


figure_3_7_correlations

    Pearson correlation coefficients and p values for
    both relationships.


NOTES
-----
No plotting is performed.
"""


import numpy as np
import pandas as pd
from scipy.stats import pearsonr


# ============================================================
# 1. Constants
# ============================================================

N_SEQUENCES = 2000

SEQUENCE_LENGTH = 147

TRIM_LEFT = 17
TRIM_RIGHT = 17

CENTRAL_LENGTH = 113

POLY_MIN_RUN = 5


# ============================================================
# 2. Check required inputs
# ============================================================

if "seq" not in globals():

    raise RuntimeError(
        "seq must already exist and contain the 2,000 "
        "experimental sequences in library order."
    )


if "NFP" not in globals():

    raise RuntimeError(
        "NFP must already exist and contain the measured "
        "nucleosome formation propensity of the 2,000 "
        "sequences."
    )


seq = [
    str(sequence).upper()
    for sequence in seq
]


NFP = np.asarray(
    NFP,
    dtype=float
)


assert len(seq) == N_SEQUENCES
assert len(NFP) == N_SEQUENCES


assert all(
    len(sequence) == SEQUENCE_LENGTH
    for sequence in seq
)


assert all(
    set(sequence).issubset(
        {"A", "C", "G", "T"}
    )
    for sequence in seq
)


# ============================================================
# 3. Extract the central 113 bp
# ============================================================

central_113 = [

    sequence[
        TRIM_LEFT:
        SEQUENCE_LENGTH - TRIM_RIGHT
    ]

    for sequence in seq
]


assert all(
    len(sequence) == CENTRAL_LENGTH
    for sequence in central_113
)


# ============================================================
# 4. GC fraction
# ============================================================

def gc_fraction(sequence):
    """
    Fraction of bases that are G or C.
    """

    gc_count = (
        sequence.count("G")
        + sequence.count("C")
    )

    return (
        gc_count
        / len(sequence)
    )


GC_fraction = np.asarray([

    gc_fraction(sequence)

    for sequence in central_113

])


assert len(GC_fraction) == 2000


# ============================================================
# 5. poly(dA:dT) bases
# ============================================================

def count_poly_dAdT_bases(
    sequence,
    minimum_run=5
):
    """
    Count bases belonging to uninterrupted homopolymeric
    A or T runs of at least minimum_run bases.

    Example:

        AAAAA       -> 5 qualifying bases
        TTTTTT      -> 6 qualifying bases
        AAAAT       -> 0 qualifying bases
        ATATAT      -> 0 qualifying bases
        CAAAAAG     -> 5 qualifying bases
    """

    qualifying_bases = 0

    i = 0

    while i < len(sequence):

        base = sequence[i]


        # Only A and T can begin a qualifying tract.

        if base not in {
            "A",
            "T"
        }:

            i += 1

            continue


        # Find the end of this homopolymeric run.

        j = i + 1

        while (
            j < len(sequence)
            and sequence[j] == base
        ):

            j += 1


        run_length = (
            j - i
        )


        if run_length >= minimum_run:

            qualifying_bases += (
                run_length
            )


        i = j


    return qualifying_bases


# ============================================================
# 6. Calculate poly(dA:dT) fraction
# ============================================================

Poly_dAdT_bases = np.asarray([

    count_poly_dAdT_bases(
        sequence,
        minimum_run=POLY_MIN_RUN
    )

    for sequence in central_113

])


Poly_dAdT_fraction = (

    Poly_dAdT_bases
    / CENTRAL_LENGTH
)


assert len(
    Poly_dAdT_fraction
) == 2000


# ============================================================
# 7. Define plotting classes
# ============================================================

# The original analysis distinguishes:
#
#     positions 1-200    SELEX
#     position 801       Widom 601
#     all remaining      Other


plot_class = np.full(
    N_SEQUENCES,
    "Other",
    dtype=object
)


plot_class[
    0:200
] = "SELEX"


plot_class[
    800
] = "Widom 601"


# ============================================================
# 8. Complete Figure 3.7 data
# ============================================================

figure_3_7 = pd.DataFrame({

    "Library_position":
        np.arange(
            1,
            2001
        ),

    "Sequence":
        seq,

    "Central_113":
        central_113,

    "GC_fraction":
        GC_fraction,

    "Poly_dAdT_bases":
        Poly_dAdT_bases,

    "Poly_dAdT_fraction":
        Poly_dAdT_fraction,

    "NFP":
        NFP,

    "Plot_class":
        plot_class
})


# ============================================================
# 9. Figure 3.7a
#    GC fraction versus NFP
# ============================================================

figure_3_7a = figure_3_7[[

    "Library_position",
    "Sequence",
    "GC_fraction",
    "NFP",
    "Plot_class"

]].copy()


valid_gc = (

    np.isfinite(
        figure_3_7a[
            "GC_fraction"
        ]
    )

    &

    np.isfinite(
        figure_3_7a[
            "NFP"
        ]
    )
)


r_gc, p_gc = pearsonr(

    figure_3_7a.loc[
        valid_gc,
        "GC_fraction"
    ],

    figure_3_7a.loc[
        valid_gc,
        "NFP"
    ]
)


# ============================================================
# 10. Figure 3.7b
#     poly(dA:dT) fraction versus NFP
# ============================================================

figure_3_7b = figure_3_7[[

    "Library_position",
    "Sequence",
    "Poly_dAdT_bases",
    "Poly_dAdT_fraction",
    "NFP",
    "Plot_class"

]].copy()


valid_poly = (

    np.isfinite(
        figure_3_7b[
            "Poly_dAdT_fraction"
        ]
    )

    &

    np.isfinite(
        figure_3_7b[
            "NFP"
        ]
    )
)


r_poly, p_poly = pearsonr(

    figure_3_7b.loc[
        valid_poly,
        "Poly_dAdT_fraction"
    ],

    figure_3_7b.loc[
        valid_poly,
        "NFP"
    ]
)


# ============================================================
# 11. Correlation summary
# ============================================================

figure_3_7_correlations = pd.DataFrame({

    "Panel": [
        "Figure 3.7a",
        "Figure 3.7b"
    ],

    "Feature": [
        "GC fraction",
        "Poly(dA:dT) fraction"
    ],

    "N": [
        int(valid_gc.sum()),
        int(valid_poly.sum())
    ],

    "Pearson_r": [
        r_gc,
        r_poly
    ],

    "P_value": [
        p_gc,
        p_poly
    ]
})


# ============================================================
# 12. Useful sequence subsets
# ============================================================

figure_3_7_SELEX = (
    figure_3_7[
        figure_3_7[
            "Plot_class"
        ] == "SELEX"
    ]
    .copy()
)


figure_3_7_Widom601 = (
    figure_3_7[
        figure_3_7[
            "Plot_class"
        ] == "Widom 601"
    ]
    .copy()
)


assert len(
    figure_3_7_SELEX
) == 200


assert len(
    figure_3_7_Widom601
) == 1


# ============================================================
# 13. Final checks
# ============================================================

assert len(
    figure_3_7
) == 2000


assert (
    figure_3_7[
        "Central_113"
    ]
    .str.len()
    .eq(113)
    .all()
)


assert (
    (
        figure_3_7[
            "GC_fraction"
        ] >= 0
    )
    &
    (
        figure_3_7[
            "GC_fraction"
        ] <= 1
    )
).all()


assert (
    (
        figure_3_7[
            "Poly_dAdT_fraction"
        ] >= 0
    )
    &
    (
        figure_3_7[
            "Poly_dAdT_fraction"
        ] <= 1
    )
).all()


# ============================================================
# 14. Report results
# ============================================================

print(
    "Figure 3.7a:"
)

print(
    "GC fraction versus NFP"
)

print(
    f"Pearson r = {r_gc:.4f}"
)

print(
    f"P = {p_gc:.4g}"
)


print(
    "\nFigure 3.7b:"
)

print(
    "poly(dA:dT) fraction versus NFP"
)

print(
    f"Pearson r = {r_poly:.4f}"
)

print(
    f"P = {p_poly:.4g}"
)


# ============================================================
# OUTPUT SUMMARY
# ============================================================

# Central 113-bp regions:
#
#     central_113
#
#
# GC fractions:
#
#     GC_fraction
#
#
# Number of bases participating in qualifying
# poly(dA:dT) tracts:
#
#     Poly_dAdT_bases
#
#
# poly(dA:dT) fractions:
#
#     Poly_dAdT_fraction
#
#
# Complete Figure 3.7 data:
#
#     figure_3_7
#
#
# LEFT PANEL:
#
#     figure_3_7a
#
#
# RIGHT PANEL:
#
#     figure_3_7b
#
#
# Correlation statistics:
#
#     figure_3_7_correlations
#
#
# No plotting is performed.