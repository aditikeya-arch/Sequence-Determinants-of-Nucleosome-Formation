"""
Intrinsic cyclisability as a function of position
=================================================

PURPOSE
-------
Predict intrinsic cyclisability (C0free) along each DNA sequence
using overlapping 50-bp windows.

REQUIRED VARIABLES
------------------
seq : list of str
    DNA sequences. All sequences must have the same length L.

REQUIRED FUNCTIONS
------------------
load_model
pred

These functions are provided by the Cyclizability Prediction code:
https://github.com/codergirl1106/Cyclizability-Prediction-Website/

OUTPUT
------
c0_mat : numpy.ndarray
    Shape:
        (number of sequences, L - 50 + 1)

    Each row contains the predicted intrinsic cyclisability profile
    for one sequence.

    Column i corresponds to the 50-bp window:
        seq[i:i+50]

    Thus, for a sequence of length L, there are L - 49 predictions.
"""

import numpy as np


# ------------------------------------------------------------
# Input sequence properties
# ------------------------------------------------------------

L = len(seq[0])


# ------------------------------------------------------------
# Generate all overlapping 50-bp windows
# ------------------------------------------------------------

seqparsed = [
    s[i:i+50]
    for s in seq
    for i in range(L - 50 + 1)
]


# ------------------------------------------------------------
# Predict C0free
# ------------------------------------------------------------

c0_vals = np.array(
    pred(load_model(0), seqparsed)
)


# ------------------------------------------------------------
# Reshape predictions into one profile per sequence
# ------------------------------------------------------------

c0_mat = c0_vals.reshape(
    len(seq),
    L - 50 + 1
)