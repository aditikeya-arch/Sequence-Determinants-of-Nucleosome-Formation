
"""
Created on Sun Oct  4 15:07:01 2026

@author: nqtf84
"""

"""
Code 13 figure 2-18 GG dinucleotide Fourier spctrum.py
======================================================

WHAT THIS SCRIPT DOES
---------------------
Generates the data underlying Figure 2.18.

Figure 2.18 shows the baseline-corrected Fourier power spectrum
associated with periodic occurrence of the GG dinucleotide among
the most enriched Round 6 SELEX sequences.

For each sequence:

    1. The central 113-bp variable region is retained.

    2. A GG indicator trace of length 113 is constructed:

           1 = GG starts at this position
           0 = otherwise

       There are 112 possible dinucleotide start positions in a
       113-bp sequence. Therefore positions 0-111 contain the
       dinucleotide-start indicators and the final element remains 0.

    3. The mean of each individual indicator trace is subtracted.

    4. A Hann window of length 113 is applied.

    5. The real FFT (rFFT) is calculated.

    6. Power is calculated as the squared magnitude of the FFT.

    7. Power is normalized by:

           L^2 * mean(Hann_window^2)

       where L = 113.

    8. The zero-frequency component is excluded.

    9. Individual power spectra are averaged across sequences.

   10. The spectrum is restricted to periodicities between
       1 and 25 bp.

   11. A smooth baseline is calculated using a
       Savitzky-Golay filter with:

           window length = 9
           polynomial order = 2

   12. The baseline is subtracted from the mean spectrum.


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

For the Round 6 dataset this filtering gives 9,803 sequences.


OUTPUTS
-------
figure_2_18

    DataFrame containing:

        Periodicity_bp
        Mean_power
        Baseline
        Baseline_corrected_power

The dataframe contains the 1-25 bp periodicity region used for
Figure 2.18.


No plotting is performed.
"""


import numpy as np
import pandas as pd

from scipy.signal import savgol_filter


# ============================================================
# 1. Analysis settings
# ============================================================

INPUT_FILE = "R6.csv"

TOP_N = 10_000

MIN_HAMMING_DISTANCE = 10

DINUCLEOTIDE = "GG"


# Central variable region of the 147-bp SELEX sequence.

TRIM = 17

RAW_LENGTH = 147

L = 113


# Periodicity range retained for Figure 2.18.

PERIODICITY_MIN = 1.0

PERIODICITY_MAX = 25.0


# Savitzky-Golay baseline parameters.

BASELINE_WINDOW_LENGTH = 9

BASELINE_POLYORDER = 2


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
    set(df.columns)
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
# 5. Select and filter the top sequences
# ============================================================

def filter_top_sequences(
    dataframe,
    top_n=TOP_N,
    min_distance=MIN_HAMMING_DISTANCE
):

    """
    Rank sequences by Counts Per Million and initially
    select the top N.

    Sequences are then processed from highest to lowest
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


# The expected Round 6 result is 9,803 sequences.

if len(
    df_top_unique
) != 9803:

    print(
        "\nWARNING:"
        "\nThe top-sequence filtering did not produce "
        "exactly 9,803 sequences."
        "\nCheck that R6.csv corresponds to the dataset "
        "used for this analysis."
    )


# ============================================================
# 6. Extract the central 113-bp sequences
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
    len(sequence) == L
    for sequence in sequences
):

    raise ValueError(
        "Central sequence extraction did not produce "
        "113-bp sequences."
    )


N = len(
    sequences
)


# ============================================================
# 7. Construct GG indicator traces
# ============================================================

# Each trace contains 113 elements.
#
# The first 112 positions correspond to the 112 possible
# dinucleotide start positions in a 113-bp sequence.
#
# The final element remains zero.

gg_indicator = np.zeros(
    (
        N,
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

        if (
            sequence[
                position:
                position + 2
            ]
            ==
            DINUCLEOTIDE
        ):

            gg_indicator[
                sequence_index,
                position
            ] = 1.0


# ============================================================
# 8. Mean-centre each individual sequence
# ============================================================

gg_centered = (

    gg_indicator

    -

    gg_indicator.mean(
        axis=1,
        keepdims=True
    )
)


# ============================================================
# 9. Hann window
# ============================================================

hann_window = np.hanning(
    L
).astype(
    np.float64
)


window_power_correction = np.mean(
    hann_window ** 2
)


gg_windowed = (

    gg_centered

    *

    hann_window[
        None,
        :
    ]
)


# ============================================================
# 10. Fourier transform
# ============================================================

fft_values = np.fft.rfft(
    gg_windowed,
    axis=1
)


# ============================================================
# 11. Normalized power spectra
# ============================================================

power = (

    np.abs(
        fft_values
    ) ** 2

    /

    (
        L ** 2
        *
        window_power_correction
    )
)


# ============================================================
# 12. Remove zero-frequency component
# ============================================================

frequency_indices = np.arange(
    power.shape[1]
)


keep = (
    frequency_indices
    >
    0
)


frequency_indices_nonzero = (
    frequency_indices[
        keep
    ]
)


periodicity = (

    L

    /

    frequency_indices_nonzero
)


power_nonzero = (
    power[
        :,
        keep
    ]
)


# ============================================================
# 13. Average individual power spectra
# ============================================================

# Each retained sequence contributes equally to the
# top-sequence analysis.

mean_power = np.mean(
    power_nonzero,
    axis=0
)


# ============================================================
# 14. Restrict spectrum to 1-25 bp periodicity
# ============================================================

periodicity_mask = (

    (periodicity >= PERIODICITY_MIN)

    &

    (periodicity <= PERIODICITY_MAX)
)


periodicity_1_25 = (
    periodicity[
        periodicity_mask
    ]
)


mean_power_1_25 = (
    mean_power[
        periodicity_mask
    ]
)


# ============================================================
# 15. Smooth baseline
# ============================================================

baseline = savgol_filter(

    mean_power_1_25,

    window_length=
        BASELINE_WINDOW_LENGTH,

    polyorder=
        BASELINE_POLYORDER
)


# ============================================================
# 16. Baseline-corrected spectrum
# ============================================================

baseline_corrected_power = (

    mean_power_1_25

    -

    baseline
)


# ============================================================
# 17. Figure 2.18 output
# ============================================================

figure_2_18 = pd.DataFrame({

    "Periodicity_bp":
        periodicity_1_25,

    "Mean_power":
        mean_power_1_25,

    "Baseline":
        baseline,

    "Baseline_corrected_power":
        baseline_corrected_power,
})


# ============================================================
# OUTPUT SUMMARY
# ============================================================

# figure_2_18
#
# contains the data required for Figure 2.18:
#
#     Periodicity_bp
#     Mean_power
#     Baseline
#     Baseline_corrected_power
#
# The baseline-corrected power is calculated as:
#
#     mean power spectrum
#     -
#     Savitzky-Golay smooth baseline
#
# using:
#
#     window_length = 9
#     polyorder     = 2
#
# No plotting is performed.