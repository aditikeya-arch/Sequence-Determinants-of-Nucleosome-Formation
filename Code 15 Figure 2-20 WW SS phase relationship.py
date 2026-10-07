


"""
Code 15 Figure 2-20 WW SS phase relationship.py
================================================

WHAT THIS SCRIPT DOES
---------------------
Generates the data underlying Figure 2.20.

Figure 2.20 examines the phase relationship between
WW and SS dinucleotide oscillations in SELEX-enriched
sequences and native yeast nucleosomal sequences.

WW dinucleotides are:

    AA, AT, TA, TT

SS dinucleotides are:

    CC, CG, GC, GG

For each sequence:

    1. The central 113-bp region is retained.

    2. Independent WW and SS indicator traces are
       constructed.

    3. Each indicator trace is mean-centred.

    4. A Hann window is applied.

    5. The Fourier transform is calculated.

    6. The Fourier component whose periodicity is closest
       to 10 bp is selected.

    7. The phase difference is calculated as:

           phase_WW - phase_SS

       and wrapped to [-pi, pi].

Figure 2.20a shows the phase difference after recentering
around perfect anti-phase:

    phase deviation =
        wrap(phase difference - pi)

The phase deviation is divided by pi, so:

     0     = perfect WW-SS anti-phase

    -1/+1  = WW and SS are in phase


Figure 2.20b quantifies the concentration of phase
differences using the circular resultant length:

    R = abs(mean(exp(i * phase_difference)))

where:

    R = 0
        phases are uniformly distributed

    R = 1
        all sequences have the same WW-SS phase
        relationship


SEQUENCE SETS
-------------
SELEX:

    The first 9,750 sequences from each already-filtered
    top-sequence dataframe are used:

        df_top_unique_R1
        df_top_unique_R2
        df_top_unique_R3
        df_top_unique_R4
        df_top_unique_R5
        df_top_unique_R6

    These dataframes contain the most enriched sequences
    after near-duplicate filtering.


Native yeast:

    Native nucleosomes are ranked by NCP score and the
    top 9,750 sequences are used.


VARIABLES THAT MUST ALREADY EXIST
---------------------------------
df_top_unique_R1
df_top_unique_R2
df_top_unique_R3
df_top_unique_R4
df_top_unique_R5
df_top_unique_R6

native_nucleosomes


OTHER SCRIPTS THAT MUST BE RUN FIRST
------------------------------------
Code 5 Native yeast nucleosomal sequences.py

The filtered top-sequence dataframes must also already
have been generated using the same top-sequence filtering
used in the preceding SELEX analyses.


OUTPUTS
-------
phase_results

    Dictionary containing the raw WW-SS phase differences
    for each sequence set.


figure_2_20a

    Long-format DataFrame containing the anti-phase-centred
    phase deviations used for panel a.


figure_2_20a_histogram

    Histogram-ready data using 60 bins spanning -1 to +1.


figure_2_20b

    Circular resultant length for R1-R6 and Top Yeast.


No plotting is performed.
"""


import numpy as np
import pandas as pd


# ============================================================
# 1. Analysis settings
# ============================================================

N_SEQUENCES = 9750


RAW_LENGTH = 147

TRIM = 17

L = 113


WW = {
    "AA",
    "AT",
    "TA",
    "TT",
}


SS = {
    "CC",
    "CG",
    "GC",
    "GG",
}


# ============================================================
# 2. Check required SELEX dataframes
# ============================================================

SELEX_DATAFRAMES = {}


for round_number in range(
    1,
    7
):

    variable_name = (
        f"df_top_unique_R{round_number}"
    )


    if variable_name not in globals():

        raise RuntimeError(

            f"{variable_name} is not defined. "
            "The filtered top-sequence dataframes "
            "must be generated before running this script."
        )


    SELEX_DATAFRAMES[
        f"R{round_number}"
    ] = globals()[
        variable_name
    ]


# ============================================================
# 3. Check native nucleosome dataframe
# ============================================================

if "native_nucleosomes" not in globals():

    raise RuntimeError(

        "native_nucleosomes is not defined. "
        "Run "
        "'Code 5 Native yeast nucleosomal sequences.py' "
        "before this script."
    )


# ============================================================
# 4. Extract the SELEX sequence sets
# ============================================================

selex_sequence_sets = {}


for round_name, dataframe in (
    SELEX_DATAFRAMES.items()
):

    if "Sequence" not in dataframe.columns:

        raise ValueError(

            f"{round_name} dataframe does not contain "
            "a Sequence column."
        )


    if len(
        dataframe
    ) < N_SEQUENCES:

        raise ValueError(

            f"{round_name} contains only "
            f"{len(dataframe)} sequences. "
            f"At least {N_SEQUENCES} are required."
        )


    # Use the first 9,750 sequences from the already
    # filtered top-sequence dataframe.

    sequences = (

        dataframe[
            "Sequence"
        ]

        .astype(str)

        .str.upper()

        .iloc[
            :N_SEQUENCES
        ]

        .tolist()
    )


    selex_sequence_sets[
        round_name
    ] = sequences


# ============================================================
# 5. Select the top 9,750 native yeast nucleosomes
# ============================================================

native = (
    native_nucleosomes
    .copy()
)


if "Sequence" not in native.columns:

    raise ValueError(

        "native_nucleosomes does not contain "
        "a Sequence column."
    )


native[
    "Sequence"
] = (

    native[
        "Sequence"
    ]

    .astype(str)

    .str.upper()
)


# Code 5 stores the native nucleosome score as NCP_score.
#
# The original Figure 2.20 analysis ranked the native
# nucleosomes by the score column in descending order.

if "NCP_score" not in native.columns:

    raise ValueError(

        "native_nucleosomes does not contain "
        "the NCP_score column."
    )


native = (

    native

    .sort_values(
        "NCP_score",
        ascending=False
    )

    .reset_index(
        drop=True
    )
)


if len(
    native
) < N_SEQUENCES:

    raise ValueError(

        "There are fewer than 9,750 native "
        "nucleosome sequences."
    )


top_yeast_sequences = (

    native[
        "Sequence"
    ]

    .iloc[
        :N_SEQUENCES
    ]

    .tolist()
)


# ============================================================
# 6. Combine all Figure 2.20 sequence sets
# ============================================================

sequence_sets = {

    "R1":
        selex_sequence_sets[
            "R1"
        ],

    "R2":
        selex_sequence_sets[
            "R2"
        ],

    "R3":
        selex_sequence_sets[
            "R3"
        ],

    "R4":
        selex_sequence_sets[
            "R4"
        ],

    "R5":
        selex_sequence_sets[
            "R5"
        ],

    "R6":
        selex_sequence_sets[
            "R6"
        ],

    "Top Yeast":
        top_yeast_sequences,
}


# ============================================================
# 7. Extract central 113-bp regions
# ============================================================

def extract_central_113(
    sequences
):

    """
    Extract positions 17:130 from each 147-bp sequence.
    """

    cores = []


    for sequence in sequences:

        sequence = str(
            sequence
        ).upper()


        if len(
            sequence
        ) != RAW_LENGTH:

            raise ValueError(

                "Figure 2.20 requires 147-bp "
                f"sequences. Found length {len(sequence)}."
            )


        core = sequence[
            TRIM:
            RAW_LENGTH - TRIM
        ]


        if len(
            core
        ) != L:

            raise RuntimeError(

                "Central sequence extraction did "
                "not produce 113 bp."
            )


        cores.append(
            core
        )


    return cores


# ============================================================
# 8. Construct WW and SS indicator traces
# ============================================================

def make_ww_ss_traces(
    sequences
):

    """
    Construct WW and SS indicator traces.

    Each trace has length 113.

    Dinucleotide starts occur at positions 0-111.
    Position 112 therefore remains zero.
    """

    n_sequences = len(
        sequences
    )


    ww_mask = np.zeros(

        (
            n_sequences,
            L
        ),

        dtype=np.float64
    )


    ss_mask = np.zeros(

        (
            n_sequences,
            L
        ),

        dtype=np.float64
    )


    for sequence_index, sequence in enumerate(
        sequences
    ):

        for position in range(
            L - 1
        ):

            dinucleotide = sequence[
                position:
                position + 2
            ]


            if dinucleotide in WW:

                ww_mask[
                    sequence_index,
                    position
                ] = 1.0


            if dinucleotide in SS:

                ss_mask[
                    sequence_index,
                    position
                ] = 1.0


    return (
        ww_mask,
        ss_mask
    )


# ============================================================
# 9. Fourier settings
# ============================================================

# Hann window used in the phase analysis.

window = np.hanning(
    L
)


# Non-zero Fourier components.

k_values = np.arange(
    1,
    L // 2 + 1
)


periodicities = (

    L

    /

    k_values
)


# Fourier component with periodicity closest to 10 bp.

k0 = k_values[

    np.argmin(

        np.abs(
            periodicities
            -
            10
        )
    )
]


selected_periodicity = (

    L

    /

    k0
)


print(
    "Selected Fourier component:",
    k0
)


print(
    "Corresponding periodicity:",
    selected_periodicity,
    "bp"
)


# For L = 113:
#
#     k0 = 11
#
#     periodicity = 113 / 11
#                 = 10.2727... bp


# ============================================================
# 10. Calculate WW-SS phase differences
# ============================================================

def calculate_phase_differences(
    full_sequences
):

    """
    Calculate the WW-SS phase difference for every
    sequence.

    Phase difference:

        phi_WW - phi_SS

    wrapped to:

        [-pi, pi]

    Sequences for which either WW or SS has exactly zero
    Fourier amplitude at the selected component are
    assigned NaN.
    """

    sequences = extract_central_113(
        full_sequences
    )


    (
        ww_mask,
        ss_mask

    ) = make_ww_ss_traces(
        sequences
    )


    # --------------------------------------------------------
    # Mean subtraction
    # --------------------------------------------------------

    ww_centered = (

        ww_mask

        -

        ww_mask.mean(
            axis=1,
            keepdims=True
        )
    )


    ss_centered = (

        ss_mask

        -

        ss_mask.mean(
            axis=1,
            keepdims=True
        )
    )


    # --------------------------------------------------------
    # Hann window
    # --------------------------------------------------------

    ww_windowed = (

        ww_centered

        *

        window[
            None,
            :
        ]
    )


    ss_windowed = (

        ss_centered

        *

        window[
            None,
            :
        ]
    )


    # --------------------------------------------------------
    # Fourier transform
    # --------------------------------------------------------

    F_ww = np.fft.rfft(

        ww_windowed,

        axis=1
    )


    F_ss = np.fft.rfft(

        ss_windowed,

        axis=1
    )


    # --------------------------------------------------------
    # Complex Fourier component nearest 10 bp
    # --------------------------------------------------------

    Z_ww = F_ww[
        :,
        k0
    ]


    Z_ss = F_ss[
        :,
        k0
    ]


    # --------------------------------------------------------
    # Exclude zero-amplitude cases
    # --------------------------------------------------------

    valid = (

        (
            np.abs(
                Z_ww
            )
            >
            0
        )

        &

        (
            np.abs(
                Z_ss
            )
            >
            0
        )
    )


    phase_difference = np.full(

        len(
            sequences
        ),

        np.nan,

        dtype=np.float64
    )


    # --------------------------------------------------------
    # Phase difference
    #
    #     phi_WW - phi_SS
    #
    # np.angle(exp(i*x)) wraps x onto [-pi, pi].
    # --------------------------------------------------------

    phi_ww = np.angle(
        Z_ww[
            valid
        ]
    )


    phi_ss = np.angle(
        Z_ss[
            valid
        ]
    )


    phase_difference[
        valid
    ] = np.angle(

        np.exp(

            1j

            *

            (
                phi_ww
                -
                phi_ss
            )
        )
    )


    return phase_difference


# ============================================================
# 11. Calculate all sequence sets
# ============================================================

phase_results = {}


for dataset, sequences in sequence_sets.items():

    print(
        "\nCalculating:",
        dataset
    )


    phase_results[
        dataset
    ] = calculate_phase_differences(
        sequences
    )


    print(

        "Valid phase measurements:",

        np.sum(

            np.isfinite(

                phase_results[
                    dataset
                ]
            )
        )
    )


# ============================================================
# 12. Figure 2.20a:
#     deviation from perfect anti-phase
# ============================================================

figure_2_20a_rows = []


for dataset, phase_difference in (
    phase_results.items()
):

    valid_phase = phase_difference[

        np.isfinite(
            phase_difference
        )
    ]


    # Perfect anti-phase corresponds to:
    #
    #     phase difference = pi
    #
    # Recenter so that anti-phase becomes zero.

    phase_deviation = np.angle(

        np.exp(

            1j

            *

            (
                valid_phase
                -
                np.pi
            )
        )
    )


    # Express deviation in units of pi.
    #
    # Therefore:
    #
    #      0 = anti-phase
    #
    #     +/-1 = in-phase

    phase_deviation_over_pi = (

        phase_deviation

        /

        np.pi
    )


    for value in phase_deviation_over_pi:

        figure_2_20a_rows.append({

            "Dataset":
                dataset,

            "Phase_deviation_radians":
                value
                *
                np.pi,

            "Phase_deviation_over_pi":
                value,
        })


figure_2_20a = pd.DataFrame(
    figure_2_20a_rows
)


# ============================================================
# 13. Histogram-ready data for Figure 2.20a
# ============================================================

# Original histogram specification:
#
#     60 bins
#
# spanning:
#
#     -1 to +1
#
# in units of pi.


HISTOGRAM_EDGES = np.linspace(

    -1,

    1,

    61
)


figure_2_20a_histogram_rows = []


for dataset in sequence_sets:

    values = (

        figure_2_20a

        .loc[
            figure_2_20a[
                "Dataset"
            ] == dataset,
            "Phase_deviation_over_pi"
        ]

        .to_numpy()
    )


    histogram_density, edges = np.histogram(

        values,

        bins=HISTOGRAM_EDGES,

        density=True
    )


    bin_centres = (

        edges[:-1]

        +

        edges[1:]

    ) / 2


    for centre, density in zip(

        bin_centres,

        histogram_density
    ):

        figure_2_20a_histogram_rows.append({

            "Dataset":
                dataset,

            "Bin_centre":
                centre,

            "Density":
                density,
        })


figure_2_20a_histogram = pd.DataFrame(
    figure_2_20a_histogram_rows
)


# ============================================================
# 14. Figure 2.20b:
#     circular resultant length
# ============================================================

def circular_resultant_length(
    phase_difference
):

    """
    Calculate:

        R = abs(mean(exp(i * phase)))

    using only finite phase measurements.
    """

    valid_phase = phase_difference[

        np.isfinite(
            phase_difference
        )
    ]


    if len(
        valid_phase
    ) == 0:

        return np.nan


    z = np.exp(

        1j

        *

        valid_phase
    )


    return np.abs(

        np.mean(
            z
        )
    )


figure_2_20b_rows = []


for dataset, phase_difference in (
    phase_results.items()
):

    resultant_length = circular_resultant_length(
        phase_difference
    )


    figure_2_20b_rows.append({

        "Dataset":
            dataset,

        "Resultant_length":
            resultant_length,

        "N_sequences":
            len(
                phase_difference
            ),

        "N_valid":
            np.sum(
                np.isfinite(
                    phase_difference
                )
            ),
    })


figure_2_20b = pd.DataFrame(
    figure_2_20b_rows
)


# ============================================================
# 15. Add round number for SELEX datasets
# ============================================================

round_lookup = {

    "R1": 1,

    "R2": 2,

    "R3": 3,

    "R4": 4,

    "R5": 5,

    "R6": 6,

    "Top Yeast": np.nan,
}


figure_2_20b[
    "Round"
] = (

    figure_2_20b[
        "Dataset"
    ]

    .map(
        round_lookup
    )
)


# ============================================================
# 16. Useful summary
# ============================================================

print(
    "\nFigure 2.20b:"
)


print(

    figure_2_20b

    .to_string(
        index=False
    )
)


# ============================================================
# OUTPUT SUMMARY
# ============================================================

# Raw phase differences:
#
#     phase_results
#
# Dictionary containing one array for each dataset:
#
#     R1
#     R2
#     R3
#     R4
#     R5
#     R6
#     Top Yeast
#
#
# Each array contains:
#
#     phi_WW - phi_SS
#
# wrapped to:
#
#     [-pi, pi]
#
#
# ------------------------------------------------------------
# Figure 2.20a
# ------------------------------------------------------------
#
#     figure_2_20a
#
# contains individual sequence phase deviations from
# perfect anti-phase.
#
#
#     figure_2_20a_histogram
#
# contains the density histogram using:
#
#     60 bins
#
#     range = -1 to +1
#
# where the x-axis is expressed in units of pi.
#
#
# Interpretation:
#
#     0
#         perfect WW-SS anti-phase
#
#     +/-1
#         WW and SS are in phase
#
#
# ------------------------------------------------------------
# Figure 2.20b
# ------------------------------------------------------------
#
#     figure_2_20b
#
# contains the circular resultant length:
#
#     R = abs(mean(exp(i * phase_difference)))
#
#
# Interpretation:
#
#     R = 0
#         broadly/uniformly distributed phase relationship
#
#     R = 1
#         identical phase relationship across sequences
#
#
# ------------------------------------------------------------
# Exact Fourier component
# ------------------------------------------------------------
#
# Sequence length:
#
#     113 bp
#
# Fourier index:
#
#     k = 11
#
# Periodicity:
#
#     113 / 11
#
#     = 10.2727... bp
#
#
# A Hann window is applied before the Fourier transform.
#
#
# ------------------------------------------------------------
# Sequence numbers
# ------------------------------------------------------------
#
# Exactly 9,750 sequences are selected for:
#
#     R1
#     R2
#     R3
#     R4
#     R5
#     R6
#     Top Yeast
#
#
# No plotting is performed.