
"""
SCRIPT NAME:
Code Figure 4-1 trinucleotide mechanical effects.py

WHAT THIS SCRIPT DOES:

Calculates the sequence-context-averaged mechanical effect of each of the
64 possible DNA trinucleotides, using the pretrained C0free neural-network
model, following the analysis in the original script `untitled0.py`.

For each trinucleotide, 10,000 independent 50-bp DNA sequences are created:

    24 randomly selected bases + fixed trinucleotide + 23 random bases

The fixed trinucleotide therefore occupies Python positions 24:27 (bases
25–27 in one-based coordinates). The other 47 bases are sampled independently
and uniformly from A, C, G and T. The pretrained C0free model predicts a
Bendability Quotient (BQ) for every sequence. The mean BQ over the 10,000
random contexts is calculated for each trinucleotide.

The relative Bendability Quotient (RBQ) is defined as:

    RBQ(trinucleotide) = mean BQ(trinucleotide) - (-0.15)

The reference value -0.15 is the baseline used in the original analysis.
It is retained here rather than recalculated or inferred from these data.

FIGURE 4.1 DATA:

The script writes a complete table of the 64 trinucleotides, their mean
predicted BQ, and their relative BQ, in ascending RBQ order. It also saves
bar charts for mean BQ and relative BQ, as in the original script.

IMPORTANT SCOPE NOTE:

The supplied `untitled0.py` establishes the 64-trinucleotide calculation.
It does not establish every panel or element of the final thesis Figure 4.1.
Accordingly, this script reproduces the trinucleotide-mechanical-effect
analysis and its underlying data; it does not invent other Figure 4.1 panels.

INPUT:

    ./adapter-free-Model/C0free

This must be the original trained Keras model (or directory) used for the
analysis. The model is not trained or modified by this script.

OUTPUT FILES (in ./figure_4_1_trinucleotide_outputs/):

    trinucleotide_mechanical_effects.csv
    results.pkl
    figure_4_1_trinucleotide_mean_BQ.png
    figure_4_1_trinucleotide_relative_BQ.png
    analysis_parameters.txt

`results.pkl` preserves the column names used in the original script, so
it can also be read by downstream analyses that expect those columns.

REPRODUCIBILITY / DIFFERENCES FROM ORIGINAL SCRIPT:

* A fixed Python random seed is supplied so that the generated sequences
  can be reproduced. The original script did not specify a seed, so its
  exact previously generated random contexts cannot be reconstructed.
* Prediction is performed in manageable chunks rather than storing all
  640,000 sequences and their one-hot representations simultaneously.
* The original one-hot encoding is retained exactly: A=0, T=1, G=2, C=3.
* The final unrelated amino-acid-range section of `untitled0.py` is omitted:
  it references an `Amino Acid` column that the trinucleotide calculation
  does not create and would therefore fail when run as supplied.
* Predicted values may also vary slightly across TensorFlow versions or
  hardware, even when the input sequences are identical.

DEPENDENCIES:

    Python 3, numpy, pandas, matplotlib, tensorflow / keras

Run from a directory containing the original `adapter-free-Model` folder:

    python "Code Figure 4-1 trinucleotide mechanical effects.py"

The script does not require data or variables from earlier Python sessions.
"""

# ===========================================================================
# 1. Imports and analysis parameters
# ===========================================================================

from itertools import product
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")  # Save figures without requiring an interactive display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras


# Original model and analysis settings.
MODEL_PATH = Path("./adapter-free-Model/C0free")
OUTPUT_DIRECTORY = Path("./figure_4_1_trinucleotide_outputs")

BASES = "ACGT"
SEQUENCE_LENGTH = 50
UPSTREAM_LENGTH = 24
TRINUCLEOTIDE_LENGTH = 3
DOWNSTREAM_LENGTH = 23
N_CONTEXTS_PER_TRINUCLEOTIDE = 10_000
BASELINE_BQ = -0.15

# A new explicit seed is used for reproducibility. The original script did
# not fix a seed, so this cannot recreate its exact original random samples.
RANDOM_SEED = 2026

# Number of sequences passed to the model per chunk, and Keras batch size.
# Chunking changes memory use but not the definition of the calculation.
PREDICTION_CHUNK_SIZE = 2_000
KERAS_BATCH_SIZE = 128

# Retain the model's original, non-alphabetical nucleotide channel order.
BASE_TO_CHANNEL = {"A": 0, "T": 1, "G": 2, "C": 3}


# ===========================================================================
# 2. Generate random DNA contexts for a specified trinucleotide
# ===========================================================================

def generate_sequences(trinucleotide, number_of_sequences, rng):
    """Generate 50-bp sequences with the same fixed central trinucleotide.

    The upstream and downstream positions are independently sampled with
    equal base probabilities. The order of calls mirrors the original code:
    first 24 upstream bases, then 23 downstream bases for each sequence.
    """
    sequences = []
    for _ in range(number_of_sequences):
        upstream = "".join(rng.choices(BASES, k=UPSTREAM_LENGTH))
        downstream = "".join(rng.choices(BASES, k=DOWNSTREAM_LENGTH))
        sequence = upstream + trinucleotide + downstream
        assert len(sequence) == SEQUENCE_LENGTH
        assert sequence[UPSTREAM_LENGTH:UPSTREAM_LENGTH + 3] == trinucleotide
        sequences.append(sequence)
    return sequences


# ===========================================================================
# 3. Encode 50-bp sequences as the C0free model expects
# ===========================================================================

def encode_sequences(sequences):
    """Return the original flattened 50 x 4 one-hot representation.

    For each DNA position, the four channels are ordered A, T, G, C. The
    resulting array has 200 float32 values per 50-bp sequence, exactly as
    in `untitled0.py` (which used `np.single`).
    """
    encoded = np.zeros((len(sequences), SEQUENCE_LENGTH * 4), dtype=np.float32)
    for sequence_index, sequence in enumerate(sequences):
        if len(sequence) != SEQUENCE_LENGTH:
            raise ValueError("Input DNA sequence does not have length 50.")
        for position, base in enumerate(sequence):
            encoded[sequence_index, position * 4 + BASE_TO_CHANNEL[base]] = 1.0
    return encoded


# ===========================================================================
# 4. Predict BQ and calculate the mean for each of the 64 trinucleotides
# ===========================================================================

def calculate_trinucleotide_effects(model):
    """Calculate mean predicted BQ and RBQ for every trinucleotide.

    Trinucleotides follow the original `itertools.product('ACGT', repeat=3)`
    order. The random generator is shared across all trinucleotides, as in
    the original script, but is now initialised from an explicit seed.

    Predictions are accumulated in float64 to avoid unnecessary numerical
    error in the mean. This is a minor numerical improvement over the
    original direct `np.mean` of its float32 predictions.
    """
    rng = random.Random(RANDOM_SEED)
    trinucleotides = ["".join(x) for x in product(BASES, repeat=3)]
    rows = []

    for index, trinucleotide in enumerate(trinucleotides, start=1):
        prediction_sum = 0.0
        number_predicted = 0

        # Each trinucleotide has 10,000 contexts, processed in smaller
        # batches so that large arrays are not held in memory at once.
        while number_predicted < N_CONTEXTS_PER_TRINUCLEOTIDE:
            chunk_size = min(
                PREDICTION_CHUNK_SIZE,
                N_CONTEXTS_PER_TRINUCLEOTIDE - number_predicted,
            )
            sequences = generate_sequences(trinucleotide, chunk_size, rng)
            encoded = encode_sequences(sequences)

            predictions = np.asarray(
                model.predict(encoded, batch_size=KERAS_BATCH_SIZE, verbose=0)
            ).reshape(-1)
            if len(predictions) != chunk_size:
                raise RuntimeError("Model returned an unexpected number of predictions.")
            if not np.all(np.isfinite(predictions)):
                raise RuntimeError(f"Non-finite predictions for {trinucleotide}.")

            prediction_sum += np.sum(predictions, dtype=np.float64)
            number_predicted += chunk_size

        mean_bq = prediction_sum / N_CONTEXTS_PER_TRINUCLEOTIDE
        relative_bq = mean_bq - BASELINE_BQ

        rows.append({
            "Trinucleotide": trinucleotide,
            "Bendability Quotient": mean_bq,
            "Relative Bendability Quotient": relative_bq,
        })
        print(
            f"[{index:02d}/64] {trinucleotide}: "
            f"mean BQ = {mean_bq:.6f}; relative BQ = {relative_bq:.6f}",
            flush=True,
        )

    results = pd.DataFrame(rows)
    assert len(results) == 64
    assert results["Trinucleotide"].is_unique

    # Preserve the original low-to-high ordering of the relative BQ plot.
    return results.sort_values(
        by="Relative Bendability Quotient", ascending=True
    ).reset_index(drop=True)


# ===========================================================================
# 5. Export numerical data for plotting and downstream analysis
# ===========================================================================

def export_results(results):
    """Save both human-readable and original-compatible result tables."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        OUTPUT_DIRECTORY / "trinucleotide_mechanical_effects.csv", index=False
    )
    results.to_pickle(OUTPUT_DIRECTORY / "results.pkl")

    parameters = (
        "Analysis: 64-trinucleotide mechanical effects for Figure 4.1\n"
        f"Model: {MODEL_PATH}\n"
        f"Bases / enumeration order: {BASES}\n"
        f"Sequence length: {SEQUENCE_LENGTH} bp\n"
        f"Fixed trinucleotide position (Python): {UPSTREAM_LENGTH}:{UPSTREAM_LENGTH + 3}\n"
        f"Upstream random bases: {UPSTREAM_LENGTH}\n"
        f"Downstream random bases: {DOWNSTREAM_LENGTH}\n"
        f"Random contexts per trinucleotide: {N_CONTEXTS_PER_TRINUCLEOTIDE}\n"
        f"Total predicted sequences: {64 * N_CONTEXTS_PER_TRINUCLEOTIDE}\n"
        f"Random seed (new; absent from original): {RANDOM_SEED}\n"
        f"Prediction chunk size: {PREDICTION_CHUNK_SIZE}\n"
        f"Keras batch size: {KERAS_BATCH_SIZE}\n"
        f"One-hot channel order: {BASE_TO_CHANNEL}\n"
        f"Baseline BQ: {BASELINE_BQ}\n"
        "Relative BQ = mean predicted BQ - baseline BQ\n"
    )
    (OUTPUT_DIRECTORY / "analysis_parameters.txt").write_text(parameters)


# ===========================================================================
# 6. Reproduce the two trinucleotide bar plots in the original script
# ===========================================================================

def plot_trinucleotide_effects(results, value_column, ylabel, filename, zero_line):
    """Plot trinucleotides in ascending relative-BQ order."""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(results["Trinucleotide"], results[value_column])
    if zero_line:
        ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Trinucleotide", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIRECTORY / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 7. Execute the complete analysis
# ===========================================================================

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find pretrained C0free model at {MODEL_PATH}. "
            "Run this script from the directory containing adapter-free-Model."
        )

    print("Loading the pretrained C0free model...", flush=True)
    model = keras.models.load_model(str(MODEL_PATH))

    print("Calculating 64 trinucleotide mean BQ values...", flush=True)
    results = calculate_trinucleotide_effects(model)
    export_results(results)

    plot_trinucleotide_effects(
        results,
        "Bendability Quotient",
        "Bendability Quotient",
        "figure_4_1_trinucleotide_mean_BQ.png",
        zero_line=False,
    )
    plot_trinucleotide_effects(
        results,
        "Relative Bendability Quotient",
        "Relative Bendability Quotient",
        "figure_4_1_trinucleotide_relative_BQ.png",
        zero_line=True,
    )

    print(f"Finished. Outputs saved in: {OUTPUT_DIRECTORY.resolve()}")


if __name__ == "__main__":
    main()
