

"""
Intrinsic cyclisability profiles across SELEX rounds
====================================================

PURPOSE
-------
For every SELEX round:

1. Predict the intrinsic cyclisability profile of every unique sequence.
2. Store these profiles in a matrix.
3. Calculate the CPM-weighted mean intrinsic cyclisability profile
   for the round.

INPUT FILES
-----------
R1.csv
R2.csv
R3.csv
R4.csv
R5.csv
R6.csv

Each CSV must contain:

    Sequence
    Counts Per Million

REQUIRED FUNCTIONS
------------------
load_model
pred

These are assumed to already be defined.

OUTPUT VARIABLES
----------------
c0_mats
    Dictionary containing the intrinsic cyclisability matrix
    for each SELEX round.

    c0_mats[r] has shape:

        (number of unique sequences in round r,
         L - 50 + 1)

mean_c0_profiles
    Dictionary containing the CPM-weighted mean intrinsic
    cyclisability profile for each round.

    mean_c0_profiles[r] has length:

        L - 50 + 1

c0_positions
    Starting positions of the 50-bp windows.

No plotting is performed.
"""

import numpy as np
import pandas as pd


# ============================================================
# Load cyclisability model once
# ============================================================

model = load_model(0)


# ============================================================
# Function for calculating cyclisability profiles
# ============================================================

def calculate_c0_profiles(sequences):

    sequences = list(sequences)

    L = len(sequences[0])

    # Check that all sequences have the same length
    if not all(len(s) == L for s in sequences):
        raise ValueError("All sequences must have the same length.")

    # Generate every overlapping 50-bp window
    seqparsed = [
        s[i:i+50]
        for s in sequences
        for i in range(L - 50 + 1)
    ]

    # Predict intrinsic cyclisability
    c0_vals = np.array(
        pred(model, seqparsed)
    )

    # One row per sequence
    c0_mat = c0_vals.reshape(
        len(sequences),
        L - 50 + 1
    )

    return c0_mat


# ============================================================
# Analyse all six rounds
# ============================================================

c0_mats = {}
mean_c0_profiles = {}
round_data = {}


for round_number in range(1, 7):

    # --------------------------------------------------------
    # Read data
    # --------------------------------------------------------

    df = pd.read_csv(f"R{round_number}.csv")

    sequences = df["Sequence"].astype(str).tolist()

    cpm = df["Counts Per Million"].to_numpy(dtype=float)


    # --------------------------------------------------------
    # Predict cyclisability profile of every unique sequence
    # --------------------------------------------------------

    c0_mat = calculate_c0_profiles(sequences)

    c0_mats[round_number] = c0_mat


    # --------------------------------------------------------
    # Convert CPM to relative abundance
    # --------------------------------------------------------

    weights = cpm / np.sum(cpm)


    # --------------------------------------------------------
    # CPM-weighted mean cyclisability profile
    # --------------------------------------------------------

    mean_profile = np.average(
        c0_mat,
        axis=0,
        weights=weights
    )

    mean_c0_profiles[round_number] = mean_profile


    # --------------------------------------------------------
    # Keep sequence/CPM information associated with matrix rows
    # --------------------------------------------------------

    round_data[round_number] = df


# ============================================================
# Position coordinates
# ============================================================

# All rounds are assumed to contain sequences of equal length.

L = len(round_data[1]["Sequence"].iloc[0])

c0_positions = np.arange(L - 50 + 1)