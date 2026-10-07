

"""
Sequence diversity and abundance inequality across SELEX rounds
===============================================================

PURPOSE
-------
Calculate the data used for Figure 2.7 and Table 2.1.

For each SELEX round this script calculates:

1. Number of unique sequences
2. Simpson effective diversity (Neff)
3. Fraction of total reads represented by the top 100 sequences
4. Lorenz curve coordinates
5. Gini coefficient

INPUT FILES
-----------
R1.csv
R2.csv
R3.csv
R4.csv
R5.csv
R6.csv

Each CSV must contain the columns:

    Sequence
    Counts Per Million

OUTPUT VARIABLES
----------------
summary
    DataFrame containing one row per SELEX round with:
        Round
        Unique sequences
        Simpson Neff
        Top 100 share
        Gini coefficient

lorenz_curves
    Dictionary containing the x and y coordinates of the
    Lorenz curve for each SELEX round.

No plotting is performed by this script.
"""

import numpy as np
import pandas as pd


# ============================================================
# Functions
# ============================================================

def analyse_round(filename):
    
    df = pd.read_csv(filename)

    # CPM values
    cpm = df["Counts Per Million"].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Convert abundance to a probability distribution
    # --------------------------------------------------------

    p = cpm / np.sum(cpm)


    # --------------------------------------------------------
    # Number of unique sequences
    # --------------------------------------------------------

    n_unique = len(df)


    # --------------------------------------------------------
    # Simpson effective diversity
    #
    # Simpson concentration:
    #     D = sum(p_i^2)
    #
    # Effective diversity:
    #     Neff = 1/D
    # --------------------------------------------------------

    simpson_concentration = np.sum(p**2)

    simpson_neff = 1 / simpson_concentration


    # --------------------------------------------------------
    # Top 100 share
    #
    # Fraction of total abundance represented by the
    # 100 most abundant unique sequences.
    # --------------------------------------------------------

    top100_share = np.sort(p)[-100:].sum()


    # --------------------------------------------------------
    # Lorenz curve
    #
    # Sort sequences from lowest to highest abundance.
    # --------------------------------------------------------

    p_sorted = np.sort(p)

    cumulative_abundance = np.cumsum(p_sorted)

    # Include the origin (0, 0)
    lorenz_y = np.concatenate(([0], cumulative_abundance))

    lorenz_x = np.linspace(
        0,
        1,
        len(lorenz_y)
    )


    # --------------------------------------------------------
    # Gini coefficient
    #
    # Gini = area between equality line and Lorenz curve
    #        divided by area under equality line.
    #
    # Since area under equality line = 0.5:
    #
    # Gini = 1 - 2 * area_under_Lorenz_curve
    # --------------------------------------------------------

    area_under_lorenz = np.trapezoid(
        lorenz_y,
        lorenz_x
    )

    gini = 1 - 2 * area_under_lorenz


    return {
        "n_unique": n_unique,
        "simpson_neff": simpson_neff,
        "top100_share": top100_share,
        "gini": gini,
        "lorenz_x": lorenz_x,
        "lorenz_y": lorenz_y
    }


# ============================================================
# Analyse all six SELEX rounds
# ============================================================

results = {}

for round_number in range(1, 7):

    filename = f"R{round_number}.csv"

    results[round_number] = analyse_round(filename)


# ============================================================
# Summary data
#
# This contains the numerical data for:
#
# Figure 2.7a -> Simpson Neff
# Figure 2.7c -> Gini coefficient
# Table 2.1  -> all four statistics
# ============================================================

summary = pd.DataFrame({

    "Round": range(1, 7),

    "Unique sequences": [
        results[r]["n_unique"]
        for r in range(1, 7)
    ],

    "Simpson Neff": [
        results[r]["simpson_neff"]
        for r in range(1, 7)
    ],

    "Top 100 share": [
        results[r]["top100_share"]
        for r in range(1, 7)
    ],

    "Gini coefficient": [
        results[r]["gini"]
        for r in range(1, 7)
    ]
})


# ============================================================
# Lorenz curve data
#
# These are the data for Figure 2.7b.
#
# Example:
#
# lorenz_curves[1]["x"]
# lorenz_curves[1]["y"]
#
# gives the x and y coordinates for Round 1.
# ============================================================

lorenz_curves = {

    r: {
        "x": results[r]["lorenz_x"],
        "y": results[r]["lorenz_y"]
    }

    for r in range(1, 7)
}


# ============================================================
# Display summary statistics
# ============================================================

print(summary)