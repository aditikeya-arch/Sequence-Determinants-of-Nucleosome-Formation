
"""
Code 12 Figure 2-17 WW dinucleotide function.py
================================================

WHAT THIS SCRIPT DOES
---------------------
Generates the data underlying Figure 2.17 using the implementation
recovered from Instructions_nuc.rtf.

For every SELEX round R1-R6 it calculates the mean positional WW
fraction.  Standard WW notation is used exactly as in the recovered
code: W = A or T, hence WW = AA, AT, TA, TT.

The recovered implementation first calculates the WW indicator on the
full 147-bp read (146 dinucleotide start positions), then retains only
dinucleotides lying wholly inside the variable region.  In original
147-bp coordinates these starts are positions 18-129 inclusive.

Both averages present in the recovered code are retained:
    * CPM-weighted mean (average over the read pool)
    * unweighted mean (average over unique sequences)

INPUT FILES
-----------
R1.csv ... R6.csv
Columns:
    Sequence
    Counts Per Million

If your archaeological CSVs use the recovered header
`Counts_per_million`, change CPM_COLUMN below accordingly.

OTHER SCRIPTS THAT MUST BE RUN FIRST
------------------------------------
None.

RELATED SCRIPT
--------------
Code 10 Figure 2-15 GC content and Fourier analysis.py

OUTPUT VARIABLES
----------------
ww_mean_profiles_weighted
ww_mean_profiles_unique
figure_2_17_weighted
figure_2_17_unique
figure_2_17              # alias of the CPM-weighted table

NO PLOTTING IS PERFORMED.
"""

import numpy as np
import pandas as pd

ROUND_NAMES = ["R1", "R2", "R3", "R4", "R5", "R6"]
SEQUENCE_COLUMN = "Sequence"
CPM_COLUMN = "Counts Per Million"
EXPECTED_LENGTH = 147
VAR_START = 18          # 1-based coordinate in the full 147-bp read
VAR_END = 130           # final variable-region base, inclusive


def ww_indicator_full(sequence):
    """Return the 146-position WW-start indicator for one 147-bp read."""
    sequence = str(sequence).upper()
    if len(sequence) != EXPECTED_LENGTH:
        raise ValueError(
            f"Expected a {EXPECTED_LENGTH}-bp sequence, found {len(sequence)}."
        )

    arr = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    is_w = (arr == ord("A")) | (arr == ord("T"))
    return (is_w[:-1] & is_w[1:]).astype(float)


def mean_ww_profiles(sequences, weights):
    """Return CPM-weighted and unique-sequence mean full-length WW profiles."""
    matrix = np.asarray([ww_indicator_full(s) for s in sequences], dtype=float)
    weights = np.asarray(weights, dtype=float)
    weighted = np.average(matrix, axis=0, weights=weights)
    unique = matrix.mean(axis=0)
    return weighted, unique


# Recovered coordinate convention:
# full WW index 0 corresponds to bases 1-2.
# To retain starts 18..129, use Python indices 17..128.
i0 = VAR_START - 1
i1 = VAR_END - 1
ww_positions = np.arange(VAR_START, VAR_END)  # 18..129, length 112

assert len(ww_positions) == 112

ww_mean_profiles_weighted = {}
ww_mean_profiles_unique = {}

for round_name in ROUND_NAMES:
    print("Processing", round_name)
    df = pd.read_csv(f"{round_name}.csv")

    weighted_full, unique_full = mean_ww_profiles(
        sequences=df[SEQUENCE_COLUMN].astype(str),
        weights=df[CPM_COLUMN]
    )

    ww_mean_profiles_weighted[round_name] = weighted_full[i0:i1]
    ww_mean_profiles_unique[round_name] = unique_full[i0:i1]

    assert len(ww_mean_profiles_weighted[round_name]) == 112
    assert len(ww_mean_profiles_unique[round_name]) == 112


figure_2_17_weighted = pd.DataFrame({"Position": ww_positions})
figure_2_17_unique = pd.DataFrame({"Position": ww_positions})

for round_name in ROUND_NAMES:
    figure_2_17_weighted[round_name] = ww_mean_profiles_weighted[round_name]
    figure_2_17_unique[round_name] = ww_mean_profiles_unique[round_name]

# Figure 2.17 is treated as the read-pool/CPM-weighted profile.
figure_2_17 = figure_2_17_weighted

print("\nWW definition: W = A/T; therefore WW = AA, AT, TA, TT")
print("Dinucleotide start coordinates:", ww_positions[0], "to", ww_positions[-1])
print("Number of plotted positions:", len(ww_positions))
