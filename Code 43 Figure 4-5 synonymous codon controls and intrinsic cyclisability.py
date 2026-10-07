

"""
SCRIPT NAME:
Code 43 Figure 4-5 synonymous codon controls and intrinsic cyclisability.py


WHAT THIS SCRIPT DOES:

Reconstructs the calculations for Figure 4.5.

FIGURE 4.5a
-------------

Generates random 99-amino-acid peptide sequences and reverse
translates each peptide in two ways:

    1. Spatially varying codon usage

       At amino-acid position 1, synonymous codons are sampled
       using the observed codon frequencies at position 1 around
       native nucleosomes.

       At position 2, frequencies from position 2 are used.

       ...

       At position 99, frequencies from position 99 are used.


    2. Uniform / overall codon usage

       Synonymous codons are sampled according to their overall
       usage frequencies, independent of position.

Intrinsic cyclisability is then predicted for both sequence sets.


FIGURE 4.5b
-------------

Starts from the native 297-bp coding nucleosome sequences
constructed for Figure 4.4.

Each native sequence is translated into its 99-amino-acid
sequence and reverse translated under three null models:

    1. Random synonymous codons

       Every synonymous codon for the required amino acid has
       equal probability.


    2. Overall codon usage

       Synonymous codons are sampled according to their overall
       usage frequencies across the native coding-nucleosome
       sequence collection, independent of position.


    3. Spatially varying codon usage

       Synonymous codons are sampled using the observed
       position-specific synonymous codon frequencies calculated
       for Figure 4.4.

These three controls preserve the encoded amino-acid sequence.


THESIS SECTION / FIGURE:

Figure 4.5


VARIABLES THAT MUST ALREADY EXIST:

From:

    Code 42 Figure 4-4 positional synonymous codon usage.py

the following must exist:

    nuc_gene_seq
    genetic_code_no_stop
    aa_to_codons
    codon_usage_df


EXTERNAL FUNCTIONS THAT MUST ALREADY EXIST:

    load_model
    pred

These are the external intrinsic-cyclisability prediction
functions used throughout the thesis.


OUTPUT VARIABLES:

figure_4_5a

    Mean intrinsic-cyclisability profiles for:

        Spatially varying
        Uniform


figure_4_5b

    Mean intrinsic-cyclisability profiles for:

        Native
        Random synonymous
        Overall codon usage
        Spatially varying


Individual sequence/profile matrices are also retained.


NO PLOTTING IS PERFORMED.
"""


import numpy as np
import pandas as pd
import random


# ================================================================
# 1. Check prerequisites
# ================================================================

required_variables = [

    "nuc_gene_seq",
    "genetic_code_no_stop",
    "aa_to_codons",
    "codon_usage_df"
]


for variable_name in required_variables:

    if variable_name not in globals():

        raise NameError(
            f"{variable_name} is not defined. "
            "Run Code 42 Figure 4-4 positional synonymous "
            "codon usage.py first."
        )


if "load_model" not in globals():

    raise NameError(
        "load_model is not defined."
    )


if "pred" not in globals():

    raise NameError(
        "pred is not defined."
    )


# ================================================================
# 2. Basic constants
# ================================================================

SEQUENCE_LENGTH = 297

N_CODONS = 99

C0_WINDOW = 50

N_C0_POSITIONS = (
    SEQUENCE_LENGTH
    -
    C0_WINDOW
    +
    1
)


assert N_C0_POSITIONS == 248


# ================================================================
# 3. Reverse genetic code
# ================================================================

# aa_to_codons was already constructed in Code 42.

amino_acids = sorted(
    aa_to_codons.keys()
)


# ================================================================
# 4. Translate a 297-bp coding sequence
# ================================================================

def translate_coding_sequence(sequence):

    sequence = str(
        sequence
    ).upper()


    if len(sequence) != 297:

        raise ValueError(
            "Coding sequence must be 297 bp."
        )


    peptide = []


    for i in range(
        0,
        297,
        3
    ):

        codon = sequence[
            i:i + 3
        ]


        if codon not in genetic_code_no_stop:

            return None


        peptide.append(
            genetic_code_no_stop[
                codon
            ]
        )


    return peptide


# ================================================================
# 5. Native peptide sequences
# ================================================================

native_peptides = []


for sequence in nuc_gene_seq:

    peptide = (
        translate_coding_sequence(
            sequence
        )
    )


    if peptide is not None:

        native_peptides.append(
            peptide
        )


print(
    "Native coding sequences:",
    len(nuc_gene_seq)
)


print(
    "Native 99-aa peptide sequences:",
    len(native_peptides)
)


# ================================================================
# 6. Position-specific codon probabilities
# ================================================================
#
# These come directly from Figure 4.4.
#
# spatial_codon_probabilities[AA][position]
#
# gives:
#
#     (codon list, probability array)
# ================================================================

spatial_codon_probabilities = {}


for amino_acid in amino_acids:

    spatial_codon_probabilities[
        amino_acid
    ] = {}


    codons = (
        aa_to_codons[
            amino_acid
        ]
    )


    for position in range(
        99
    ):

        probabilities = []


        for codon in codons:

            row = codon_usage_df[
                (
                    codon_usage_df[
                        "Amino_Acid"
                    ]
                    ==
                    amino_acid
                )
                &
                (
                    codon_usage_df[
                        "Codon"
                    ]
                    ==
                    codon
                )
            ]


            value = float(
                row[
                    f"Pos{position + 1}"
                ].iloc[0]
            )


            probabilities.append(
                value
            )


        probabilities = np.asarray(
            probabilities,
            dtype=float
        )


        # --------------------------------------------------------
        # Normally these already sum to 1.
        #
        # The fallback protects against a position where the
        # amino acid was absent from the native data.
        # --------------------------------------------------------

        if (
            np.any(
                ~np.isfinite(
                    probabilities
                )
            )
            or
            probabilities.sum()
            <=
            0
        ):

            probabilities = np.ones(
                len(codons),
                dtype=float
            )


        probabilities = (
            probabilities
            /
            probabilities.sum()
        )


        spatial_codon_probabilities[
            amino_acid
        ][
            position
        ] = (
            codons,
            probabilities
        )


# ================================================================
# 7. Overall synonymous codon frequencies
# ================================================================
#
# For each amino acid, pool codon counts across all 99 positions.
#
# This gives the position-independent codon usage control.
# ================================================================

overall_codon_counts = {

    amino_acid: {

        codon: 0

        for codon in (
            aa_to_codons[
                amino_acid
            ]
        )
    }

    for amino_acid
    in amino_acids
}


for sequence in nuc_gene_seq:

    sequence = sequence.upper()


    for i in range(
        0,
        297,
        3
    ):

        codon = sequence[
            i:i + 3
        ]


        if codon not in genetic_code_no_stop:

            continue


        amino_acid = (
            genetic_code_no_stop[
                codon
            ]
        )


        overall_codon_counts[
            amino_acid
        ][
            codon
        ] += 1


overall_codon_probabilities = {}


for amino_acid in amino_acids:

    codons = (
        aa_to_codons[
            amino_acid
        ]
    )


    codon_counts = np.asarray(
        [
            overall_codon_counts[
                amino_acid
            ][
                codon
            ]

            for codon in codons
        ],
        dtype=float
    )


    if codon_counts.sum() == 0:

        probabilities = np.ones(
            len(codons),
            dtype=float
        )

    else:

        probabilities = codon_counts


    probabilities = (
        probabilities
        /
        probabilities.sum()
    )


    overall_codon_probabilities[
        amino_acid
    ] = (
        codons,
        probabilities
    )


# ================================================================
# 8. Reverse-translation functions
# ================================================================

def reverse_translate_equal(
    peptide
):

    """
    Choose uniformly among synonymous codons.
    """

    dna = []


    for amino_acid in peptide:

        codons = (
            aa_to_codons[
                amino_acid
            ]
        )


        dna.append(
            random.choice(
                codons
            )
        )


    return "".join(
        dna
    )


def reverse_translate_overall(
    peptide
):

    """
    Choose synonymous codons according to overall codon usage,
    independent of position.
    """

    dna = []


    for amino_acid in peptide:

        codons, probabilities = (
            overall_codon_probabilities[
                amino_acid
            ]
        )


        codon = random.choices(
            codons,
            weights=probabilities,
            k=1
        )[0]


        dna.append(
            codon
        )


    return "".join(
        dna
    )


def reverse_translate_spatial(
    peptide
):

    """
    Choose synonymous codons according to position-specific
    codon usage around native nucleosomes.
    """

    dna = []


    for position, amino_acid in enumerate(
        peptide
    ):

        codons, probabilities = (
            spatial_codon_probabilities[
                amino_acid
            ][
                position
            ]
        )


        codon = random.choices(
            codons,
            weights=probabilities,
            k=1
        )[0]


        dna.append(
            codon
        )


    return "".join(
        dna
    )


# ================================================================
# 9. Intrinsic-cyclisability prediction function
# ================================================================
#
# A 297-bp sequence contains:
#
#     297 - 50 + 1 = 248
#
# overlapping 50-bp prediction windows.
#
# The prediction is performed in batches to avoid generating an
# excessively large list of 50-bp sequences at once.
# ================================================================

def predict_c0_profiles(
    sequences,
    sequence_batch_size=500
):

    model = load_model(
        0
    )


    profile_batches = []


    for batch_start in range(
        0,
        len(sequences),
        sequence_batch_size
    ):

        batch = sequences[
            batch_start:
            batch_start
            +
            sequence_batch_size
        ]


        windows = [

            sequence[
                i:i + 50
            ]

            for sequence in batch

            for i in range(
                N_C0_POSITIONS
            )
        ]


        predictions = np.asarray(
            pred(
                model,
                windows
            ),
            dtype=float
        )


        profiles = predictions.reshape(
            len(batch),
            N_C0_POSITIONS
        )


        profile_batches.append(
            profiles
        )


        print(
            "Predicted:",
            min(
                batch_start
                +
                sequence_batch_size,
                len(sequences)
            ),
            "/",
            len(sequences)
        )


    return np.vstack(
        profile_batches
    )


# ================================================================
# 10. FIGURE 4.5a
# ================================================================
#
# Random peptide sequences.
#
# The historical analysis used 40,000 random peptide sequences.
#
# Each peptide contains 99 amino acids.
# ================================================================

N_RANDOM_PEPTIDES = 40000


random_peptides = [

    random.choices(
        amino_acids,
        k=99
    )

    for _ in range(
        N_RANDOM_PEPTIDES
    )
]


# ================================================================
# 11. Reverse translate random peptides
# ================================================================

random_peptide_spatial_sequences = [

    reverse_translate_spatial(
        peptide
    )

    for peptide in random_peptides
]


random_peptide_uniform_sequences = [

    reverse_translate_overall(
        peptide
    )

    for peptide in random_peptides
]


# ================================================================
# 12. Predict Figure 4.5a cyclisability profiles
# ================================================================

print()
print(
    "Figure 4.5a: spatially varying codon usage"
)


c0_45a_spatial = predict_c0_profiles(
    random_peptide_spatial_sequences
)


print()
print(
    "Figure 4.5a: overall codon usage"
)


c0_45a_uniform = predict_c0_profiles(
    random_peptide_uniform_sequences
)


# ================================================================
# 13. Mean profiles for Figure 4.5a
# ================================================================

mean_c0_45a_spatial = np.mean(
    c0_45a_spatial,
    axis=0
)


mean_c0_45a_uniform = np.mean(
    c0_45a_uniform,
    axis=0
)


# ================================================================
# 14. Cyclisability coordinates
# ================================================================
#
# Prediction index corresponds to the START of a 50-bp window.
#
# For plotting relative to the centre of the 297-bp sequence,
# use the centre of each 50-bp window.
#
# Window starts:
#
#     0 ... 247
#
# Window centres:
#
#     24.5 ... 271.5
#
# Sequence centre:
#
#     148
#
# Therefore:
#
#     -123.5 ... +123.5
# ================================================================

c0_position_bp = (

    np.arange(
        N_C0_POSITIONS
    )

    +
    24.5

    -
    148
)


figure_4_5a = pd.DataFrame(
    {
        "Position_bp":
            c0_position_bp,

        "Spatially_varying":
            mean_c0_45a_spatial,

        "Uniform":
            mean_c0_45a_uniform
    }
)


# ================================================================
# 15. FIGURE 4.5b
# ================================================================
#
# Preserve the native amino-acid sequence and create three
# synonymous controls.
# ================================================================

native_sequences_45b = []

native_peptides_45b = []


for sequence in nuc_gene_seq:

    peptide = (
        translate_coding_sequence(
            sequence
        )
    )


    if peptide is None:

        continue


    native_sequences_45b.append(
        sequence
    )


    native_peptides_45b.append(
        peptide
    )


# ================================================================
# 16. Equal-probability synonymous randomisation
# ================================================================

random_synonymous_sequences_45b = [

    reverse_translate_equal(
        peptide
    )

    for peptide
    in native_peptides_45b
]


# ================================================================
# 17. Overall codon-usage randomisation
# ================================================================

overall_usage_sequences_45b = [

    reverse_translate_overall(
        peptide
    )

    for peptide
    in native_peptides_45b
]


# ================================================================
# 18. Position-specific codon-usage randomisation
# ================================================================

spatial_usage_sequences_45b = [

    reverse_translate_spatial(
        peptide
    )

    for peptide
    in native_peptides_45b
]


# ================================================================
# 19. Predict native sequences
# ================================================================

print()
print(
    "Figure 4.5b: native"
)


c0_45b_native = predict_c0_profiles(
    native_sequences_45b
)


# ================================================================
# 20. Predict random synonymous control
# ================================================================

print()
print(
    "Figure 4.5b: random synonymous"
)


c0_45b_random = predict_c0_profiles(
    random_synonymous_sequences_45b
)


# ================================================================
# 21. Predict overall codon-usage control
# ================================================================

print()
print(
    "Figure 4.5b: overall codon usage"
)


c0_45b_overall = predict_c0_profiles(
    overall_usage_sequences_45b
)


# ================================================================
# 22. Predict spatial codon-usage control
# ================================================================

print()
print(
    "Figure 4.5b: spatially varying codon usage"
)


c0_45b_spatial = predict_c0_profiles(
    spatial_usage_sequences_45b
)


# ================================================================
# 23. Mean Figure 4.5b profiles
# ================================================================

mean_c0_45b_native = np.mean(
    c0_45b_native,
    axis=0
)


mean_c0_45b_random = np.mean(
    c0_45b_random,
    axis=0
)


mean_c0_45b_overall = np.mean(
    c0_45b_overall,
    axis=0
)


mean_c0_45b_spatial = np.mean(
    c0_45b_spatial,
    axis=0
)


# ================================================================
# 24. Figure 4.5b output
# ================================================================

figure_4_5b = pd.DataFrame(
    {
        "Position_bp":
            c0_position_bp,

        "Native":
            mean_c0_45b_native,

        "Random_synonymous":
            mean_c0_45b_random,

        "Overall_codon_usage":
            mean_c0_45b_overall,

        "Spatially_varying":
            mean_c0_45b_spatial
    }
)


# ================================================================
# 25. Final summary
# ================================================================

print()
print(
    "Figure 4.5 calculations complete."
)


print()
print(
    "Figure 4.5a random peptides:",
    N_RANDOM_PEPTIDES
)


print(
    "Figure 4.5b native sequences:",
    len(
        native_sequences_45b
    )
)


print()
print(
    "Figure 4.5a:"
)


print(
    figure_4_5a.head()
)


print()
print(
    "Figure 4.5b:"
)


print(
    figure_4_5b.head()
)