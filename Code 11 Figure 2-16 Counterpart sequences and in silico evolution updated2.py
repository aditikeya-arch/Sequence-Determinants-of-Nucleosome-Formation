

"""
Code 11 Figure 2-16 Counterpart sequences and in silico evolution.py
====================================================================

WHAT THIS SCRIPT DOES
---------------------
Generates the data underlying Figure 2.16.

Figure 2.16a
------------
Uses the filtered set of 9,803 highly enriched R6 sequences
previously used for Figures 2.10 and 2.11.

For each SELEX sequence, counterpart random sequences are
constructed while preserving specified sequence features.

The main control described explicitly in the thesis preserves
the purine/pyrimidine identity at every position:

    purine position    -> random A or G
    pyrimidine position -> random C or T

Intrinsic cyclisability profiles are calculated for:

    1. original R6 SELEX sequences
    2. purine/pyrimidine-preserving counterparts
    3. WW-position-preserving counterparts

and averaged across sequences.


Figure 2.16b
------------
For each of the 9,803 R6 sequences, an independently evolved
DNA sequence is generated whose intrinsic cyclisability
profile is selected to approach that particular SELEX
sequence's profile.

Evolution procedure:

    1. Retain the target sequence's existing 17-bp flanks and
       randomize only its central 113 bp.
    3. Generate all 339 possible single-base mutations in
       the central 113-bp region.
    4. Calculate the 98-position intrinsic cyclisability
       profile of every mutant.
    5. Calculate summed squared error (SSE) from the target profile.
    6. Accept the best mutant only if it improves on the current SSE.
    7. Repeat for 15 iterations.

Finally, GC Fourier power spectra are calculated for the
central 113 bp of:

    original R6 sequences
    evolved sequences

using the same method as Figure 2.15.


VARIABLES/FUNCTIONS THAT MUST ALREADY EXIST
-------------------------------------------
load_model
pred


OTHER SCRIPTS THAT MUST BE RUN FIRST
------------------------------------
Code 4 PCA analysis Fig 2-10.py

This script should provide the filtered set of 9,803
R6 sequences.

Because the exact variable name may differ between versions
of Code 4, this script assumes:

    r6_top_filtered

with column:

    Sequence


RECOVERED IMPLEMENTATION NOTE
-----------------------------
The original evolution code preserved the first and last 17 bp
of each target SELEX sequence directly.  No separately supplied
adapter strings are required.


OUTPUT VARIABLES
----------------
figure_2_16a
figure_2_16b

purine_pyrimidine_counterparts
ww_counterparts

evolved_sequences

selex_mean_c0_profile
purine_pyrimidine_mean_c0_profile
ww_mean_c0_profile

selex_mean_power
evolved_mean_power
"""


import numpy as np
import pandas as pd


# ============================================================
# 1. Settings
# ============================================================

N_ITERATIONS = 15

rng = np.random.default_rng(1)


# ============================================================
# 2. Adapter handling confirmed from recovered original code
# ============================================================

# The recovered evolution code did NOT require separately supplied
# adapter strings.  For each target SELEX sequence, its existing
# first 17 bp and last 17 bp were retained exactly, while only the
# central 113-bp region was randomized/mutated.

# ============================================================
# 3. Check filtered R6 sequences from Code 4
# ============================================================

if "r6_top_filtered" not in globals():

    raise RuntimeError(
        "r6_top_filtered is not defined. Run "
        "'Code 4 PCA analysis Fig 2-10.py' first."
    )


assert len(r6_top_filtered) == 9803


selex_sequences = (
    r6_top_filtered[
        "Sequence"
    ]
    .astype(str)
    .str.upper()
    .tolist()
)


assert all(
    len(sequence) == 147
    for sequence in selex_sequences
)


# ============================================================
# 4. Load cyclisability model
# ============================================================

model = load_model(0)


# ============================================================
# 5. Calculate intrinsic cyclisability profiles
# ============================================================

def calculate_c0_profiles(
    sequences,
    model
):

    """
    Calculate the intrinsic cyclisability profile of each
    sequence.

    For 147-bp sequences this produces 98 values.
    """

    sequences = list(sequences)

    L = len(sequences[0])


    if not all(
        len(sequence) == L
        for sequence in sequences
    ):

        raise ValueError(
            "All sequences must have the same length."
        )


    seqparsed = [

        sequence[i:i+50]

        for sequence in sequences

        for i in range(
            L - 50 + 1
        )
    ]


    c0_vals = np.asarray(

        pred(
            model,
            seqparsed
        ),

        dtype=float
    )


    c0_mat = c0_vals.reshape(

        len(sequences),

        L - 50 + 1
    )


    return c0_mat


# ============================================================
# 6. Original SELEX profiles
# ============================================================

print(
    "Calculating R6 SELEX cyclisability profiles..."
)


selex_c0_mat = calculate_c0_profiles(
    selex_sequences,
    model
)


assert selex_c0_mat.shape == (
    9803,
    98
)


selex_mean_c0_profile = np.mean(
    selex_c0_mat,
    axis=0
)


c0_positions = np.arange(
    25,
    25 + 98
)


# ============================================================
# 7. Purine/pyrimidine-preserving randomization
# ============================================================

def purine_pyrimidine_counterpart(
    sequence,
    rng
):

    """
    Randomize sequence while preserving purine/pyrimidine
    identity at every position.

        A/G -> random A/G
        C/T -> random C/T

    This therefore preserves the R/Y pattern exactly.
    """

    output = []


    for base in sequence:

        if base in ("A", "G"):

            output.append(
                rng.choice(
                    ["A", "G"]
                )
            )

        elif base in ("C", "T"):

            output.append(
                rng.choice(
                    ["C", "T"]
                )
            )

        else:

            raise ValueError(
                f"Unexpected base: {base}"
            )


    return "".join(output)


purine_pyrimidine_counterparts = [

    purine_pyrimidine_counterpart(
        sequence,
        rng
    )

    for sequence in selex_sequences
]


print(
    "Calculating purine/pyrimidine counterpart profiles..."
)


purine_pyrimidine_c0_mat = (
    calculate_c0_profiles(
        purine_pyrimidine_counterparts,
        model
    )
)


purine_pyrimidine_mean_c0_profile = np.mean(
    purine_pyrimidine_c0_mat,
    axis=0
)


# ============================================================
# 8. WW-position-preserving counterpart
# ============================================================

#
# RECONSTRUCTION NOTE
# -------------------
#
# The Figure 2.16 caption states that a second counterpart
# set preserved the locations of WW dinucleotides.
#
# The thesis text available to us does not specify the exact
# randomization algorithm used to generate this control.
#
# The implementation below is therefore a reconstruction.
#
# Here:
#
#     WW = AA, AT, TA, TT
#
# Positions that participate in a WW dinucleotide are kept
# A/T, while all other positions are randomized.
#
# If the original code is recovered, replace this function
# with the original algorithm.
#


def ww_preserving_counterpart(
    sequence,
    rng
):

    sequence = sequence.upper()

    L = len(sequence)


    ww_position = np.zeros(
        L,
        dtype=bool
    )


    # Identify bases participating in an original WW pair.

    for i in range(
        L - 1
    ):

        if (
            sequence[i] in ("A", "T")
            and
            sequence[i+1] in ("A", "T")
        ):

            ww_position[i] = True
            ww_position[i+1] = True


    output = []


    for i in range(L):

        if ww_position[i]:

            output.append(
                rng.choice(
                    ["A", "T"]
                )
            )

        else:

            output.append(
                rng.choice(
                    ["A", "C", "G", "T"]
                )
            )


    return "".join(output)


ww_counterparts = [

    ww_preserving_counterpart(
        sequence,
        rng
    )

    for sequence in selex_sequences
]


print(
    "Calculating WW counterpart profiles..."
)


ww_c0_mat = calculate_c0_profiles(
    ww_counterparts,
    model
)


ww_mean_c0_profile = np.mean(
    ww_c0_mat,
    axis=0
)


# ============================================================
# 9. Figure 2.16a output
# ============================================================

figure_2_16a = pd.DataFrame({

    "Position":
        c0_positions,

    "SELEX":
        selex_mean_c0_profile,

    "Purine_pyrimidine_control":
        purine_pyrimidine_mean_c0_profile,

    "WW_control":
        ww_mean_c0_profile
})


# ============================================================
# 10. Random starting counterpart used for evolution
# ============================================================

def random_dna(length, rng):
    return "".join(rng.choice(["A", "C", "G", "T"], size=length))


def make_random_evolution_start(target_sequence, rng):
    """
    Recovered original behaviour:

    retain target_sequence[:17] and target_sequence[130:] exactly,
    and replace only the central 113 bp with random DNA.
    """
    target_sequence = str(target_sequence).upper()
    if len(target_sequence) != 147:
        raise ValueError("Expected a 147-bp target sequence.")
    return target_sequence[:17] + random_dna(113, rng) + target_sequence[130:]


# ============================================================
# 11. Generate all 339 single-base mutants
# ============================================================

def all_single_mutants(sequence_147):
    """Generate all 339 point mutants in positions 17:130 only."""
    sequence_147 = str(sequence_147).upper()
    if len(sequence_147) != 147:
        raise ValueError("Expected a 147-bp sequence.")

    bases = ("A", "C", "G", "T")
    left = sequence_147[:17]
    core = list(sequence_147[17:130])
    right = sequence_147[130:]
    mutants = []

    for i in range(113):
        original = core[i]
        for new_base in bases:
            if new_base == original:
                continue
            new_core = core.copy()
            new_core[i] = new_base
            mutants.append(left + "".join(new_core) + right)

    assert len(mutants) == 339
    return mutants


# ============================================================
# 12. Loss used in the recovered original implementation
# ============================================================

def squared_error_loss(profiles, target_profile):
    """Return sum of squared errors across the 98 profile positions."""
    return ((profiles - target_profile[None, :]) ** 2).sum(axis=1)


# ============================================================
# 13. Evolve one sequence to match one target profile
# ============================================================

def evolve_to_profile(target_sequence, target_profile, model, rng, n_iterations=15):
    """
    Recovered original greedy evolution algorithm.

    Start with random central 113 bp while retaining the target's
    existing 17-bp flanks.  At each iteration evaluate all 339
    single mutants.  Crucially, accept the best mutant ONLY if its
    SSE is lower than the current sequence's SSE.  Thus an iteration
    can leave the sequence unchanged.
    """
    current_sequence = make_random_evolution_start(target_sequence, rng)
    current_profile = calculate_c0_profiles([current_sequence], model)[0]
    current_loss = ((current_profile - target_profile) ** 2).sum()

    loss_history = []

    for _ in range(n_iterations):
        mutants = all_single_mutants(current_sequence)
        mutant_profiles = calculate_c0_profiles(mutants, model)
        mutant_losses = squared_error_loss(mutant_profiles, target_profile)

        best_index = int(np.argmin(mutant_losses))
        best_loss = float(mutant_losses[best_index])

        if best_loss < current_loss:
            current_sequence = mutants[best_index]
            current_profile = mutant_profiles[best_index]
            current_loss = best_loss

        loss_history.append(current_loss)

    return current_sequence, current_profile, float(current_loss), np.asarray(loss_history)


# ============================================================
# 14. Notes on thesis wording versus recovered code
# ============================================================

# The thesis describes RMS deviation.  The recovered implementation
# minimized sum of squared errors.  Since every candidate profile has
# the same 98 positions, SSE and RMS rank candidates identically; this
# script nevertheless preserves the recovered SSE implementation.

# ============================================================
# 15. Evolve one sequence for every R6 target profile
# ============================================================

# WARNING: this is a very large computation.  The recovered original
# implementation was run as a SLURM array job and batched model
# predictions.  The loop below preserves the algorithm in a single
# archaeological script; for production use, parallelize/batch it.

evolved_sequences = []
evolved_profiles = []
evolution_final_losses = []
evolution_loss_histories = []

for sequence_index in range(len(selex_sequences)):
    print("Evolving sequence", sequence_index + 1, "of", len(selex_sequences))

    target_sequence = selex_sequences[sequence_index]
    target_profile = selex_c0_mat[sequence_index]

    evolved_sequence, evolved_profile, final_loss, loss_history = evolve_to_profile(
        target_sequence=target_sequence,
        target_profile=target_profile,
        model=model,
        rng=rng,
        n_iterations=N_ITERATIONS
    )

    evolved_sequences.append(evolved_sequence)
    evolved_profiles.append(evolved_profile)
    evolution_final_losses.append(final_loss)
    evolution_loss_histories.append(loss_history)

evolved_profiles = np.asarray(evolved_profiles)
evolution_final_losses = np.asarray(evolution_final_losses)
evolution_loss_histories = np.asarray(evolution_loss_histories)

# ============================================================
# 16. Save evolved sequences
# ============================================================

#
# Strongly recommended because generating these sequences
# is computationally expensive.
#

evolved_sequences_df = pd.DataFrame({

    "Target_SELEX_sequence":
        selex_sequences,

    "Evolved_sequence":
        evolved_sequences,

    "Final_SSE":
        evolution_final_losses
})


evolved_sequences_df.to_csv(
    "Figure_2_16_evolved_sequences.csv",
    index=False
)


# ============================================================
# 17. GC Fourier power spectrum
#
# Same method as Code 10 Figure 2-15 GC content and
# Fourier analysis.py
# ============================================================

def gc_indicator(
    sequence
):

    return np.array(
        [
            1.0 if base in ("G", "C") else 0.0
            for base in sequence.upper()
        ],
        dtype=float
    )


def gc_power_spectrum(
    sequence
):

    sequence = sequence.upper()

    L = len(sequence)


    trace = gc_indicator(
        sequence
    )


    # Mean centre.

    trace = (
        trace
        -
        np.mean(trace)
    )


    # Hann window.

    window = np.hanning(
        L
    )


    windowed_trace = (
        trace
        *
        window
    )


    # FFT.

    fft_values = np.fft.rfft(
        windowed_trace
    )


    # Power.

    power = (
        np.abs(
            fft_values
        ) ** 2
    )


    # Normalization used in Figure 2.15.

    window_correction = np.mean(
        window ** 2
    )


    power = power / (
        (L ** 2)
        *
        window_correction
    )


    # Remove zero frequency.

    power = power[1:]


    frequency_indices = np.arange(
        1,
        len(power) + 1
    )


    periodicity = (
        L
        /
        frequency_indices
    )


    return (
        periodicity,
        power
    )


# ============================================================
# 18. Central 113 bp
# ============================================================

def central_113(
    sequence
):

    if len(sequence) != 147:

        raise ValueError(
            "Expected 147-bp sequence."
        )


    return sequence[
        17:130
    ]


# ============================================================
# 19. Mean power spectrum
# ============================================================

def mean_power_spectrum(
    sequences
):

    spectra = []


    for sequence in sequences:

        periodicity, power = (
            gc_power_spectrum(
                sequence
            )
        )


        spectra.append(
            power
        )


    spectra = np.asarray(
        spectra
    )


    return (
        periodicity,
        np.mean(
            spectra,
            axis=0
        )
    )


# ============================================================
# 20. Fourier spectrum of original SELEX sequences
# ============================================================

selex_sequences_113 = [

    central_113(
        sequence
    )

    for sequence in selex_sequences
]


periodicity, selex_mean_power = (
    mean_power_spectrum(
        selex_sequences_113
    )
)


# ============================================================
# 21. Fourier spectrum of evolved sequences
# ============================================================

evolved_sequences_113 = [

    central_113(
        sequence
    )

    for sequence in evolved_sequences
]


evolved_periodicity, evolved_mean_power = (
    mean_power_spectrum(
        evolved_sequences_113
    )
)


if not np.allclose(
    periodicity,
    evolved_periodicity
):

    raise RuntimeError(
        "Fourier periodicity axes do not match."
    )


# ============================================================
# 22. Figure 2.16b output
# ============================================================

figure_2_16b = pd.DataFrame({

    "Periodicity_bp":
        periodicity,

    "SELEX":
        selex_mean_power,

    "Evolved":
        evolved_mean_power
})


figure_2_16b = (
    figure_2_16b
    .sort_values(
        "Periodicity_bp"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# FIGURE OUTPUT SUMMARY
# ============================================================

# Figure 2.16a:
#
#     figure_2_16a
#
# columns:
#
#     Position
#     SELEX
#     Purine_pyrimidine_control
#     WW_control
#
#
# Figure 2.16b:
#
#     figure_2_16b
#
# columns:
#
#     Periodicity_bp
#     SELEX
#     Evolved
#
#
# Expensive intermediate result:
#
#     Figure_2_16_evolved_sequences.csv
#
# contains the evolved sequence corresponding to each
# individual R6 target sequence.