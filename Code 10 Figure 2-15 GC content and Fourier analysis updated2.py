

"""
Code 10 Figure 2-15 GC content and Fourier analysis.py
===========================================================

WHAT THIS SCRIPT DOES
---------------------
Generates the data underlying Figure 2.15.

Figure 2.15a
------------
Left:
    Mean GC content as a function of position across the
    mapped native yeast nucleosomal sequences.

Right:
    Mean GC content as a function of position for each
    SELEX round R1-R6.

    SELEX sequences are weighted by Counts Per Million
    because the thesis describes the profile as averaged
    over all reads.


Figure 2.15b
------------
Calculates Fourier power spectra of GC-content oscillations
for individual sequences, then averages those spectra.

For each 113-bp sequence:

    1. G/C -> 1
       A/T -> 0

    2. subtract the sequence's own mean GC fraction

    3. multiply by a Hann window of length 113

    4. calculate the real FFT (rFFT)

    5. power = squared magnitude of FFT coefficient

    6. normalize by:

           sequence_length^2
           *
           mean(Hann_window^2)

    7. remove zero-frequency component

    8. convert frequency index k to periodicity:

           periodicity = 113 / k

SELEX power spectra are averaged using CPM weights.

Native yeast nucleosome power spectra are averaged without
weights.


INPUT FILES
-----------
R1.csv
R2.csv
R3.csv
R4.csv
R5.csv
R6.csv


VARIABLES THAT MUST ALREADY EXIST
---------------------------------
native_nucleosomes

This DataFrame must contain:

    Sequence

with one 147-bp sequence for each mapped native yeast
nucleosome.


OTHER SCRIPTS THAT MUST BE RUN FIRST
------------------------------------
Code 5 Native yeast nucleosomal sequences.py

This script generates:

    native_nucleosomes


OUTPUT VARIABLES
----------------
native_mean_gc_profile
selex_mean_gc_profiles

periodicity

native_mean_power_spectrum
selex_mean_power_spectra

figure_2_15a_native
figure_2_15a_selex
figure_2_15b


RECOVERED-CODE NOTE
-------------------
Instructions_nuc.rtf later confirmed the Hann/rFFT normalization used
here and showed that the working analysis also retained unweighted
(unique-sequence) profiles and a categorical GC+AT spectrum. Those
confirmed outputs are included at the end of this script.

NO PLOTTING IS PERFORMED.
"""


import numpy as np
import pandas as pd


# ============================================================
# 1. Settings
# ============================================================

round_names = [
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6"
]


# ============================================================
# 2. Check that Code 5 has been run
# ============================================================

if "native_nucleosomes" not in globals():

    raise RuntimeError(
        "native_nucleosomes is not defined. "
        "Run 'Code 5 Native yeast nucleosomal sequences.py' "
        "before running this script."
    )


# ============================================================
# 3. Extract central 113 bp
# ============================================================

def central_113(sequence):

    """
    Extract the central 113 bp from a 147-bp sequence.

    Removes 17 bp from each end.
    """

    sequence = str(sequence).upper()

    if len(sequence) != 147:

        raise ValueError(
            "Expected a 147-bp sequence, "
            f"but found length {len(sequence)}."
        )

    return sequence[17:130]


# ============================================================
# 4. Convert DNA sequence to GC indicator trace
# ============================================================

def gc_indicator(sequence):

    """
    Convert sequence into binary GC trace:

        A/T = 0
        G/C = 1
    """

    sequence = str(sequence).upper()

    return np.array(
        [
            1.0 if base in ("G", "C") else 0.0
            for base in sequence
        ],
        dtype=float
    )


# ============================================================
# 5. Mean GC profile
# ============================================================

def mean_gc_profile(
    sequences,
    weights=None
):

    """
    Calculate mean GC fraction at every sequence position.

    If weights are supplied, the position-wise average is
    weighted by those values.
    """

    gc_matrix = np.array(
        [
            gc_indicator(sequence)
            for sequence in sequences
        ],
        dtype=float
    )


    if weights is None:

        mean_profile = np.mean(
            gc_matrix,
            axis=0
        )

    else:

        mean_profile = np.average(
            gc_matrix,
            axis=0,
            weights=np.asarray(
                weights,
                dtype=float
            )
        )


    return mean_profile


# ============================================================
# 6. Load SELEX data
# ============================================================

round_data = {}


for round_name in round_names:

    print(
        "Reading",
        round_name
    )


    df = pd.read_csv(
        f"{round_name}.csv"
    )


    df["Sequence_113"] = [
        central_113(sequence)
        for sequence in df["Sequence"]
    ]


    round_data[
        round_name
    ] = df


# ============================================================
# 7. Native yeast sequences
# ============================================================

native_sequences_147 = (
    native_nucleosomes[
        "Sequence"
    ]
    .astype(str)
    .str.upper()
    .tolist()
)


#
# Figure 2.15a uses the complete 147-bp native nucleosomal
# sequences aligned at their dyads.
#

native_mean_gc_profile = mean_gc_profile(
    native_sequences_147
)


#
# Coordinates relative to nucleosome dyad.
#
# For a 147-bp sequence:
#
#     -73 ... 0 ... +73
#

native_positions = np.arange(
    -73,
    74
)


assert len(native_mean_gc_profile) == 147
assert len(native_positions) == 147


# ============================================================
# 8. Figure 2.15a:
#
# Mean positional GC profile for SELEX rounds
# ============================================================

selex_mean_gc_profiles = {}


for round_name in round_names:

    print(
        "Calculating positional GC profile:",
        round_name
    )


    df = round_data[
        round_name
    ]


    #
    # The thesis describes this as averaged over all reads.
    #
    # The CSV contains unique sequences plus CPM, therefore
    # CPM weighting reconstructs the read-weighted average.
    #

    selex_mean_gc_profiles[
        round_name
    ] = mean_gc_profile(

        sequences=df[
            "Sequence_113"
        ],

        weights=df[
            "Counts Per Million"
        ]
    )


selex_positions = np.arange(
    -56,
    57
)


assert len(selex_positions) == 113


# ============================================================
# 9. Figure 2.15a output tables
# ============================================================

figure_2_15a_native = pd.DataFrame({

    "Position":
        native_positions,

    "Mean_GC":
        native_mean_gc_profile
})


figure_2_15a_selex = pd.DataFrame({

    "Position":
        selex_positions
})


for round_name in round_names:

    figure_2_15a_selex[
        round_name
    ] = selex_mean_gc_profiles[
        round_name
    ]


# ============================================================
# 10. Fourier analysis function
# ============================================================

def gc_power_spectrum(sequence):

    """
    Calculate the normalized Fourier power spectrum of the
    GC indicator trace of one sequence.

    Procedure follows thesis Section 2.2.14.
    """

    sequence = str(sequence).upper()

    L = len(sequence)


    # --------------------------------------------------------
    # Binary GC trace
    # --------------------------------------------------------

    trace = gc_indicator(
        sequence
    )


    # --------------------------------------------------------
    # Mean-centre each individual sequence
    # --------------------------------------------------------

    trace = (
        trace
        -
        np.mean(trace)
    )


    # --------------------------------------------------------
    # Hann window
    # --------------------------------------------------------

    window = np.hanning(
        L
    )


    windowed_trace = (
        trace
        *
        window
    )


    # --------------------------------------------------------
    # Real FFT
    # --------------------------------------------------------

    fft_values = np.fft.rfft(
        windowed_trace
    )


    # --------------------------------------------------------
    # Power
    # --------------------------------------------------------

    power = (
        np.abs(fft_values) ** 2
    )


    # --------------------------------------------------------
    # Normalize power
    #
    # Thesis:
    #
    # divide by sequence length squared and by a correction
    # factor equal to mean squared value of Hann window.
    # --------------------------------------------------------

    window_correction = np.mean(
        window ** 2
    )


    power = power / (
        (L ** 2)
        *
        window_correction
    )


    # --------------------------------------------------------
    # Remove zero-frequency component
    # --------------------------------------------------------

    power = power[1:]


    # --------------------------------------------------------
    # Periodicity
    #
    # For FFT index k:
    #
    #     periodicity = L / k
    # --------------------------------------------------------

    frequency_indices = np.arange(
        1,
        len(power) + 1
    )


    periodicity = (
        L
        /
        frequency_indices
    )


    return periodicity, power


# ============================================================
# 11. Mean power spectrum for a collection of sequences
# ============================================================

def mean_gc_power_spectrum(
    sequences,
    weights=None
):

    """
    Calculate power spectrum independently for every
    sequence and then average the individual spectra.

    If weights are supplied, spectra are averaged using
    those weights.
    """

    sequences = list(
        sequences
    )


    all_power = []


    for sequence in sequences:

        sequence_periodicity, power = (
            gc_power_spectrum(
                sequence
            )
        )


        all_power.append(
            power
        )


    all_power = np.asarray(
        all_power,
        dtype=float
    )


    if weights is None:

        mean_power = np.mean(
            all_power,
            axis=0
        )

    else:

        mean_power = np.average(
            all_power,
            axis=0,
            weights=np.asarray(
                weights,
                dtype=float
            )
        )


    return (
        sequence_periodicity,
        mean_power
    )


# ============================================================
# 12. Native nucleosome Fourier spectrum
# ============================================================

#
# The thesis states that the Fourier analysis was repeated
# for 113-bp fragments spanning the mapped native yeast
# nucleosomes.
#
# Therefore trim the native 147-bp nucleosomal sequences to
# the central 113 bp before Fourier analysis.
#

native_sequences_113 = [
    central_113(sequence)
    for sequence in native_sequences_147
]


print(
    "Calculating native yeast GC power spectrum..."
)


periodicity, native_mean_power_spectrum = (
    mean_gc_power_spectrum(
        native_sequences_113
    )
)


# ============================================================
# 13. SELEX Fourier spectra
# ============================================================

selex_mean_power_spectra = {}


for round_name in round_names:

    print(
        "Calculating GC power spectrum:",
        round_name
    )


    df = round_data[
        round_name
    ]


    round_periodicity, mean_power = (
        mean_gc_power_spectrum(

            sequences=df[
                "Sequence_113"
            ],

            weights=df[
                "Counts Per Million"
            ]
        )
    )


    #
    # All sequences are 113 bp, so all rounds must have
    # identical periodicity coordinates.
    #

    if not np.allclose(
        round_periodicity,
        periodicity
    ):

        raise RuntimeError(
            "Periodicity coordinates differ between datasets."
        )


    selex_mean_power_spectra[
        round_name
    ] = mean_power


# ============================================================
# 14. Figure 2.15b output
# ============================================================

figure_2_15b = pd.DataFrame({

    "Periodicity_bp":
        periodicity,

    "Native_yeast":
        native_mean_power_spectrum
})


for round_name in round_names:

    figure_2_15b[
        round_name
    ] = selex_mean_power_spectra[
        round_name
    ]


# ============================================================
# 15. Sort by periodicity for convenient plotting
# ============================================================

#
# FFT naturally produces periodicities from long to short:
#
#     113, 56.5, 37.7, ...
#
# Sorting is convenient if plotting periodicity on the x-axis.
#

figure_2_15b_sorted = (
    figure_2_15b
    .sort_values(
        "Periodicity_bp"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 16. Useful sanity checks
# ============================================================

print(
    "\nNumber of native nucleosomal sequences:",
    len(native_sequences_147)
)


print(
    "Native GC profile length:",
    len(native_mean_gc_profile)
)


print(
    "SELEX GC profile length:",
    len(
        selex_mean_gc_profiles["R1"]
    )
)


print(
    "Number of non-zero Fourier frequencies:",
    len(periodicity)
)


# Find Fourier bins nearest 10 bp and 3 bp.

index_10bp = np.argmin(
    np.abs(
        periodicity - 10
    )
)


index_3bp = np.argmin(
    np.abs(
        periodicity - 3
    )
)


print(
    "\nFourier bin nearest 10 bp:",
    periodicity[index_10bp]
)


print(
    "Fourier bin nearest 3 bp:",
    periodicity[index_3bp]
)


print(
    "\nNative power near 10 bp:",
    native_mean_power_spectrum[
        index_10bp
    ]
)


print(
    "Native power near 3 bp:",
    native_mean_power_spectrum[
        index_3bp
    ]
)


# ============================================================
# FIGURE OUTPUT SUMMARY
# ============================================================

# Figure 2.15a LEFT:
#
#     figure_2_15a_native
#
# columns:
#
#     Position
#     Mean_GC
#
#
# Figure 2.15a RIGHT:
#
#     figure_2_15a_selex
#
# columns:
#
#     Position
#     R1
#     R2
#     R3
#     R4
#     R5
#     R6
#
#
# Figure 2.15b:
#
#     figure_2_15b_sorted
#
# columns:
#
#     Periodicity_bp
#     Native_yeast
#     R1
#     R2
#     R3
#     R4
#     R5
#     R6
# ============================================================
# 17. RECOVERED ORIGINAL-CODE OUTPUTS
# ============================================================
#
# Instructions_nuc.rtf recovered after this script was first
# reconstructed confirms the Fourier normalization above exactly.
# It also shows that the working analysis retained BOTH CPM-weighted
# and unique-sequence (unweighted) averages, and used a "categorical"
# spectrum equal to GC power + AT power.  Because AT is the exact
# complement of GC, after per-sequence mean-centering Pat == Pgc and
# the categorical spectrum is exactly 2 * GC power (apart from floating
# point roundoff).  We calculate it explicitly here to preserve the
# recovered implementation rather than relying on that identity.


def categorical_gc_at_power_spectrum(sequence):
    """Recovered implementation: GC-indicator power + AT-indicator power."""
    sequence = str(sequence).upper()
    L = len(sequence)

    is_gc = np.array([1.0 if b in ("G", "C") else 0.0 for b in sequence])
    is_at = np.array([1.0 if b in ("A", "T") else 0.0 for b in sequence])

    gc0 = is_gc - is_gc.mean()
    at0 = is_at - is_at.mean()

    window = np.hanning(L)
    U = np.mean(window ** 2)

    Xgc = np.fft.rfft(gc0 * window)
    Xat = np.fft.rfft(at0 * window)

    Pgc = (np.abs(Xgc) ** 2) / (L ** 2 * U)
    Pat = (np.abs(Xat) ** 2) / (L ** 2 * U)

    k = np.arange(L // 2 + 1)
    keep = k > 0
    return L / k[keep], (Pgc + Pat)[keep]


def mean_categorical_gc_at_power_spectrum(sequences, weights=None):
    spectra = []
    p_axis = None
    for sequence in sequences:
        this_axis, this_power = categorical_gc_at_power_spectrum(sequence)
        if p_axis is None:
            p_axis = this_axis
        spectra.append(this_power)

    spectra = np.asarray(spectra, dtype=float)
    if weights is None:
        mean_power = spectra.mean(axis=0)
    else:
        mean_power = np.average(spectra, axis=0, weights=np.asarray(weights, dtype=float))
    return p_axis, mean_power


# Positional SELEX GC profiles: recovered code retained weighted and unique.
selex_mean_gc_profiles_unique = {}
for round_name in round_names:
    df = round_data[round_name]
    selex_mean_gc_profiles_unique[round_name] = mean_gc_profile(
        sequences=df["Sequence_113"],
        weights=None
    )

figure_2_15a_selex_unique = pd.DataFrame({"Position": selex_positions})
for round_name in round_names:
    figure_2_15a_selex_unique[round_name] = selex_mean_gc_profiles_unique[round_name]


# Recovered categorical spectra for SELEX: weighted and unique.
selex_categorical_power_weighted = {}
selex_categorical_power_unique = {}

for round_name in round_names:
    df = round_data[round_name]

    recovered_periodicity, power_w = mean_categorical_gc_at_power_spectrum(
        df["Sequence_113"],
        weights=df["Counts Per Million"]
    )
    _, power_u = mean_categorical_gc_at_power_spectrum(
        df["Sequence_113"],
        weights=None
    )

    if not np.allclose(recovered_periodicity, periodicity):
        raise RuntimeError("Recovered categorical periodicity axis differs from GC axis.")

    selex_categorical_power_weighted[round_name] = power_w
    selex_categorical_power_unique[round_name] = power_u

figure_2_15b_categorical_weighted = pd.DataFrame({
    "Periodicity_bp": recovered_periodicity
})
figure_2_15b_categorical_unique = pd.DataFrame({
    "Periodicity_bp": recovered_periodicity
})

for round_name in round_names:
    figure_2_15b_categorical_weighted[round_name] = selex_categorical_power_weighted[round_name]
    figure_2_15b_categorical_unique[round_name] = selex_categorical_power_unique[round_name]

figure_2_15b_categorical_weighted = (
    figure_2_15b_categorical_weighted.sort_values("Periodicity_bp").reset_index(drop=True)
)
figure_2_15b_categorical_unique = (
    figure_2_15b_categorical_unique.sort_values("Periodicity_bp").reset_index(drop=True)
)

print("\nRecovered-code check:")
print("  Fourier normalization confirmed: |rFFT|^2 / (L^2 * mean(Hann^2))")
print("  k=0 excluded; periodicity = 113/k")
print("  CPM-weighted and unique-sequence outputs retained")
print("  categorical spectrum = GC power + AT power")
