#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT NAME:
Code 45 Figure 4-3 usage frequency weighted mean mechanical effect.py

WHAT THIS SCRIPT DOES:

Calculates the positional, synonymous-codon-usage-frequency-weighted mean
mechanical effect for each of the 20 standard amino acids, using results
already calculated by the preceding two thesis analysis scripts:

    Code 41 Figure 4-1 trinucleotide mechanical effects.py
    Code 42 Figure 4-2 positional synonymous codon usage.py

The calculation follows the corresponding section of `untitled12.py`.
The script DOES NOT recalculate trinucleotide mechanical effects, extract
nucleosome-associated coding sequences, or recount synonymous codons.

For amino acid a at codon position p, the plotted quantity is:

    weighted_mean_RBQ(a, p) = SUM over c in synonymous_codons(a)
                               [frequency(c, p) * RBQ(c)]

Here frequency(c, p) is the position-specific usage frequency of codon c
among codons encoding amino acid a, and RBQ(c) is the relative Bendability
Quotient of the same three-base codon measured in Figure 4.1. The
frequencies for synonymous codons of a given amino acid sum to one when
that amino acid is represented at the position.

This is a codon-usage-weighted MEAN, not a total over the abundance of an
amino acid at a position. It therefore describes the codon preferences
conditional on the amino acid, rather than the amino-acid composition of
the nucleosome-associated coding sequences.

EXPECTED EXECUTION ENVIRONMENT:

Run this script in the SAME Python session / notebook namespace AFTER the
Figure 4.1 and Figure 4.2 scripts have been run. For example, in IPython:

    %run -i "Code 41 Figure 4-1 trinucleotide mechanical effects.py"
    %run -i "Code 42 Figure 4-2 positional synonymous codon usage.py"
    %run -i "Code 45 Figure 4-3 usage frequency weighted mean mechanical effect.py"

The `-i` option is important: it executes the scripts using the current
interactive namespace. Running the files as three separate `python` shell
commands does NOT preserve variables between processes.

REQUIRED VARIABLES ALREADY IN MEMORY:

    results
        pandas DataFrame with columns:
            'Trinucleotide'
            'Relative Bendability Quotient'
        One RBQ for each of the 64 DNA trinucleotides. If Code 41
        produced the DataFrame but did not assign it to `results`, assign
        its result DataFrame to that variable before running this code.

    codon_usage_df
        pandas DataFrame produced by Code 42, with columns:
            'Codon', 'Pos1', ..., 'Pos99'
        An optional 'Amino_Acid' column is accepted and ignored here.

    aa_to_codons  OR  aa_codons
        Dictionary mapping each amino-acid one-letter code to its list
        of synonymous codons. Code 42 uses `aa_to_codons`, whereas the
        original `untitled12.py` used `aa_codons`.

OUTPUT VARIABLES:

    rbq_lookup
        Dictionary mapping trinucleotide strings to their RBQ values.

    weighted_mean_rbq_df
        Wide table with one row per amino acid and Pos1...Pos99 columns.

    figure_4_3_data
        Long-format table with one row per amino acid and position,
        including 'Amino_Acid', 'Codon_position', 'Position_bp',
        'Weighted_mean_RBQ', and 'Frequency_sum'.

OUTPUT FILES (in ./figure_4_3_outputs/):

    figure_4_3_weighted_mean_mechanical_effect.csv
    figure_4_3_weighted_mean_mechanical_effect_wide.csv
    figure_4_3_weighted_mean_mechanical_effect.png
    figure_4_3_weighted_mean_mechanical_effect.pdf

The 5 x 4 panel arrangement, alphabetical amino-acid order, 99 codon
positions, and line plots follow `untitled12.py`. Files are saved to make
Figure 4.3 reproducible without rerunning the full upstream analyses.

MISSING DATA:

In `untitled12.py`, a missing codon frequency is skipped, starting the
weighted sum at zero. Accordingly, if ALL synonymous-codon frequencies
for an amino acid and position are missing, the original script would
plot zero. This script preserves that convention for numerical
compatibility, while also recording the frequency sum in the output
so such positions can be recognised. It issues a warning if any
amino-acid/position has an incomplete frequency sum.

No randomisation or neural-network inference is performed in this script.
"""

# =====================================================================
# 1. Imports and output directory
# =====================================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIRECTORY = Path('figure_4_3_outputs')
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
NUMBER_OF_CODON_POSITIONS = 99


# =====================================================================
# 2. Check that the upstream scripts have populated the namespace
# =====================================================================
#
# Unlike a standalone analysis, this script intentionally uses the
# existing Python variables. It should not silently load a potentially
# different copy of a results file from disk.
# =====================================================================

required_names = ('results', 'codon_usage_df')
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise RuntimeError(
        'Required upstream variables are missing: '
        + ', '.join(missing_names)
        + '. Run Codes 41 and 42 in this Python session first.'
    )

if 'aa_to_codons' in globals():
    amino_acid_codons = aa_to_codons
elif 'aa_codons' in globals():
    amino_acid_codons = aa_codons
else:
    raise RuntimeError(
        "Expected 'aa_to_codons' (Code 42) or 'aa_codons' "
        '(untitled12.py) in the current namespace.'
    )

if not isinstance(results, pd.DataFrame):
    raise TypeError("'results' must be a pandas DataFrame.")
if not isinstance(codon_usage_df, pd.DataFrame):
    raise TypeError("'codon_usage_df' must be a pandas DataFrame.")

required_rbq_columns = {'Trinucleotide', 'Relative Bendability Quotient'}
if not required_rbq_columns.issubset(results.columns):
    raise ValueError(
        'The Figure 4.1 results DataFrame must contain: '
        + ', '.join(sorted(required_rbq_columns))
    )

position_columns = [f'Pos{position}' for position in range(1, 100)]
required_usage_columns = {'Codon', *position_columns}
if not required_usage_columns.issubset(codon_usage_df.columns):
    raise ValueError('The Figure 4.2 codon_usage_df lacks required columns.')


# =====================================================================
# 3. Construct and validate the trinucleotide RBQ lookup
# =====================================================================
#
# Each synonymous codon is also a DNA trinucleotide. Consequently, the
# context-averaged RBQ measured for each trinucleotide in Figure 4.1 can
# be assigned directly to the corresponding codon.
# =====================================================================

rbq_table = results[['Trinucleotide', 'Relative Bendability Quotient']].copy()
rbq_table['Trinucleotide'] = rbq_table['Trinucleotide'].astype(str).str.upper()
rbq_table['Relative Bendability Quotient'] = pd.to_numeric(
    rbq_table['Relative Bendability Quotient'], errors='raise'
)

if rbq_table['Trinucleotide'].duplicated().any():
    raise ValueError('Duplicate trinucleotides found in Figure 4.1 results.')
if rbq_table['Relative Bendability Quotient'].isna().any():
    raise ValueError('Missing RBQ values found in Figure 4.1 results.')

rbq_lookup = dict(zip(
    rbq_table['Trinucleotide'],
    rbq_table['Relative Bendability Quotient']
))


# =====================================================================
# 4. Prepare and validate the positional synonymous-codon frequencies
# =====================================================================

usage_table = codon_usage_df[['Codon', *position_columns]].copy()
usage_table['Codon'] = usage_table['Codon'].astype(str).str.upper()

if usage_table['Codon'].duplicated().any():
    raise ValueError('Duplicate codons found in codon_usage_df.')

for column in position_columns:
    usage_table[column] = pd.to_numeric(usage_table[column], errors='raise')

frequency_lookup = usage_table.set_index('Codon')
amino_acids = sorted(amino_acid_codons)

if len(amino_acids) != 20:
    warnings.warn(
        f'Expected 20 standard amino acids; found {len(amino_acids)}.',
        RuntimeWarning
    )

all_codons = {
    str(codon).upper()
    for aa in amino_acids
    for codon in amino_acid_codons[aa]
}
missing_rbq = sorted(all_codons.difference(rbq_lookup))
missing_frequency = sorted(all_codons.difference(frequency_lookup.index))

if missing_rbq:
    raise ValueError('RBQ values missing for codons: ' + ', '.join(missing_rbq))
if missing_frequency:
    raise ValueError(
        'Positional frequencies missing for codons: '
        + ', '.join(missing_frequency)
    )


# =====================================================================
# 5. Calculate codon-usage-weighted mean RBQ at each position
# =====================================================================
#
# This is the numerical calculation performed in the corresponding block
# of untitled12.py:
#
#     weighted_mean = 0
#     for codon in synonymous_codons:
#         if frequency is not NaN:
#             weighted_mean += frequency * RBQ(codon)
#
# We retain that missing-value behaviour exactly. In addition, we
# record the sum of the observed synonymous-codon frequencies as a
# diagnostic of the calculation at each amino-acid/position pair.
# =====================================================================

wide_rows = []
long_rows = []
incomplete_frequency_sums = []

for amino_acid in amino_acids:
    synonymous_codons = [
        str(codon).upper() for codon in amino_acid_codons[amino_acid]
    ]
    weighted_means = np.zeros(NUMBER_OF_CODON_POSITIONS, dtype=float)

    for position_index in range(NUMBER_OF_CODON_POSITIONS):
        column = f'Pos{position_index + 1}'
        weighted_value = 0.0
        frequency_sum = 0.0

        for codon in synonymous_codons:
            frequency = frequency_lookup.at[codon, column]
            mechanical_effect = rbq_lookup[codon]

            if not np.isnan(frequency):
                weighted_value += frequency * mechanical_effect
                frequency_sum += frequency

        weighted_means[position_index] = weighted_value

        if not np.isclose(frequency_sum, 1.0, atol=1e-6, rtol=0):
            incomplete_frequency_sums.append((
                amino_acid, position_index + 1, frequency_sum
            ))

        long_rows.append({
            'Amino_Acid': amino_acid,
            'Codon_position': position_index + 1,
            # Code 42 uses codon 50 as the zero coordinate.
            'Position_bp': (position_index + 1 - 50) * 3,
            'Weighted_mean_RBQ': weighted_value,
            'Frequency_sum': frequency_sum,
        })

    wide_row = {'Amino_Acid': amino_acid}
    wide_row.update({
        f'Pos{position + 1}': weighted_means[position]
        for position in range(NUMBER_OF_CODON_POSITIONS)
    })
    wide_rows.append(wide_row)

weighted_mean_rbq_df = pd.DataFrame(wide_rows)
figure_4_3_data = pd.DataFrame(long_rows)

if incomplete_frequency_sums:
    warnings.warn(
        f'{len(incomplete_frequency_sums)} amino-acid/position pairs '
        'have synonymous-codon frequencies that do not sum to 1. '
        'Their weighted values follow the original untitled12.py '
        'missing-frequency convention; inspect Frequency_sum in the CSV.',
        RuntimeWarning
    )

assert len(figure_4_3_data) == len(amino_acids) * 99


# =====================================================================
# 6. Export all numerical data used to generate Figure 4.3
# =====================================================================

long_path = OUTPUT_DIRECTORY / 'figure_4_3_weighted_mean_mechanical_effect.csv'
wide_path = OUTPUT_DIRECTORY / 'figure_4_3_weighted_mean_mechanical_effect_wide.csv'

figure_4_3_data.to_csv(long_path, index=False)
weighted_mean_rbq_df.to_csv(wide_path, index=False)


# =====================================================================
# 7. Plot the positional mechanical-effect profiles for 20 amino acids
# =====================================================================
#
# As in untitled12.py, the amino acids are arranged alphabetically in
# five rows and four columns. Each panel displays the usage-weighted
# mean RBQ across the 99 codon positions surrounding the nucleosome.
#
# The axis labels are intentionally kept consistent with the historical
# plot ('Region number' and 'Weighted mean RBQ').
# =====================================================================

fig, axes = plt.subplots(5, 4, figsize=(18, 20), sharex=True)
axes = axes.flatten()

for axis, amino_acid in zip(axes, amino_acids):
    row = weighted_mean_rbq_df.loc[
        weighted_mean_rbq_df['Amino_Acid'] == amino_acid,
        position_columns
    ].iloc[0]

    axis.plot(
        range(1, NUMBER_OF_CODON_POSITIONS + 1),
        row.to_numpy(dtype=float),
        linewidth=2
    )
    axis.set_title(amino_acid)
    axis.set_xlabel('Region number')
    axis.set_ylabel('Weighted mean RBQ')
    axis.grid(alpha=0.3)

for axis in axes[len(amino_acids):]:
    axis.remove()

fig.tight_layout()
fig.savefig(
    OUTPUT_DIRECTORY / 'figure_4_3_weighted_mean_mechanical_effect.png',
    dpi=300, bbox_inches='tight'
)
fig.savefig(
    OUTPUT_DIRECTORY / 'figure_4_3_weighted_mean_mechanical_effect.pdf',
    bbox_inches='tight'
)
plt.show()


# =====================================================================
# 8. Final checks and output summary
# =====================================================================

print('\nFigure 4.3 weighted mean mechanical-effect data ready.')
print('Number of amino acids:', len(amino_acids))
print('Codon positions per amino acid:', NUMBER_OF_CODON_POSITIONS)
print('Total plotted values:', len(figure_4_3_data))
print('Positions with frequency sums differing from 1:',
      len(incomplete_frequency_sums))
print('Output directory:', OUTPUT_DIRECTORY.resolve())
print('\nPreview:')
print(figure_4_3_data.head())
