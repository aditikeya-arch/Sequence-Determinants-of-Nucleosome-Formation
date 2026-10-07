

"""
Code 24 Chapter 3 construction of sequences 802 to 2000.py
===========================================================

WHAT THIS SCRIPT DOES
---------------------
Constructs the 1,199 sequences occupying positions
802-2000 of the 2,000-member competitive nucleosome
reconstitution library.

The sequences are assembled in the following order:

    s1    100 native yeast nucleosomal coding sequences
    s2    100 sequences designed to mimic the intrinsic
          cyclisability profiles of the s1 sequences

    s3     20 sequences designed to match a prescribed
           Gaussian intrinsic-cyclisability profile

    s4     20 sequences designed to match the intrinsic
           cyclisability profile of native nucleosome #510

    s5     20 sequences designed to match the intrinsic
           cyclisability profile of native nucleosome #545

    s6    110 synonymous codon-randomized variants of
          native coding nucleosome #345

    s7      1 original native sequence #345

    s8    828 random DNA sequences

Total:

    100 + 100 + 20 + 20 + 20 + 110 + 1 + 828
    = 1,199 sequences


EXTERNAL FUNCTIONS THAT MUST ALREADY EXIST
------------------------------------------
load_model

pred

These are the intrinsic-cyclisability prediction
functions used throughout the thesis.


INPUT DATA THAT MUST ALREADY EXIST
----------------------------------

native_coding_sequences

    List of 297-bp native yeast nucleosomal coding
    sequences.

    The ordering must correspond to the ordering used
    when the experimental library was designed.

    At minimum, this list must contain sequences with
    indices 345 and 500-599.


native_coding_proteins

    List of corresponding 99-amino-acid protein
    sequences.

    native_coding_proteins[i] must be the translation
    of native_coding_sequences[i].

    Sequence index 345 is used for the synonymous
    codon-randomization experiment.


ASSUMPTIONS
-----------
Python uses zero-based indexing.

Therefore:

    native sequence #345 -> index 345

in the original analysis code.

Likewise:

    native sequences 500-599

are selected using:

    native_coding_sequences[500:600]


All sequences synthesized for the 147-bp experimental
library contain:

    17-bp left adapter
    113-bp variable region
    17-bp right adapter


NOTES
-----
The original construction involved several separately
saved intermediate sequence sets. Here they are generated
in a single script so that the construction logic is
explicit and reproducible.

The exact Gaussian target used for the experimental
library should be checked. The calculation supplied for
the target uses:

    peak = 0.0
    baseline = -0.15
    standard deviation = 20

although another comment describes the peak as 0.1.

This script therefore uses peak = 0.0, matching the
explicit calculation, but defines it as a parameter so
that it can be changed easily if required.

No plotting is performed.
"""


import random
import numpy as np
import pandas as pd


# ============================================================
# 1. Constants
# ============================================================

LEFT_ADAPTER = "CTTATCTCCCACCGTCC"
RIGHT_ADAPTER = "GGCAGAAGACAAGGGAA"

BASES = ("A", "C", "G", "T")

LIBRARY_LENGTH = 147
VARIABLE_LENGTH = 113

CODING_LENGTH = 297

WINDOW = 50

LIBRARY_PROFILE_LENGTH = (
    LIBRARY_LENGTH - WINDOW + 1
)  # 98

CODING_PROFILE_LENGTH = (
    CODING_LENGTH - WINDOW + 1
)  # 248


assert len(LEFT_ADAPTER) == 17
assert len(RIGHT_ADAPTER) == 17

assert (
    len(LEFT_ADAPTER)
    + VARIABLE_LENGTH
    + len(RIGHT_ADAPTER)
    == LIBRARY_LENGTH
)


# ============================================================
# 2. Reproducibility
# ============================================================

RANDOM_SEED = 0

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 3. Check required external inputs
# ============================================================

if "load_model" not in globals():
    raise RuntimeError(
        "load_model must already be defined."
    )

if "pred" not in globals():
    raise RuntimeError(
        "pred must already be defined."
    )

if "native_coding_sequences" not in globals():
    raise RuntimeError(
        "native_coding_sequences must contain the ordered "
        "297-bp native yeast nucleosomal coding sequences."
    )

if "native_coding_proteins" not in globals():
    raise RuntimeError(
        "native_coding_proteins must contain the corresponding "
        "99-aa protein sequences."
    )


assert len(native_coding_sequences) > 599
assert len(native_coding_proteins) > 345


# ============================================================
# 4. General sequence helpers
# ============================================================

def random_dna(length):
    """Generate a random DNA sequence with equal base probabilities."""

    return "".join(
        random.choice(BASES)
        for _ in range(length)
    )


def central_region(sequence, length):
    """Return the central region of the requested length."""

    start = (len(sequence) - length) // 2

    return sequence[
        start:start + length
    ]


def make_library_sequence(variable_region):
    """
    Add the common 17-bp adapters to a 113-bp
    variable region.
    """

    assert len(variable_region) == VARIABLE_LENGTH

    sequence = (
        LEFT_ADAPTER
        + variable_region
        + RIGHT_ADAPTER
    )

    assert len(sequence) == LIBRARY_LENGTH

    return sequence


def coding_to_library_sequence(sequence):
    """
    Convert a 297-bp coding-region sequence into the
    corresponding 147-bp experimental construct.

    The central 113 bp are retained and flanked by the
    common adapters.
    """

    assert len(sequence) == CODING_LENGTH

    variable_region = central_region(
        sequence,
        VARIABLE_LENGTH
    )

    return make_library_sequence(
        variable_region
    )


# ============================================================
# 5. Intrinsic-cyclisability prediction
# ============================================================

model = load_model(0)


def predict_profiles(sequences, batch_size=500_000):
    """
    Predict the complete intrinsic-cyclisability profile
    for a collection of equal-length sequences.

    A 147-bp sequence produces 98 values.

    A 297-bp sequence produces 248 values.
    """

    if len(sequences) == 0:
        return np.empty((0, 0))

    sequence_length = len(sequences[0])

    assert all(
        len(s) == sequence_length
        for s in sequences
    )

    profile_length = (
        sequence_length - WINDOW + 1
    )

    windows = [
        sequence[i:i + WINDOW]

        for sequence in sequences

        for i in range(profile_length)
    ]

    predictions = []

    for start in range(
        0,
        len(windows),
        batch_size
    ):

        batch = windows[
            start:start + batch_size
        ]

        predictions.extend(
            np.asarray(
                pred(model, batch)
            ).reshape(-1)
        )

    predictions = np.asarray(
        predictions,
        dtype=float
    )

    return predictions.reshape(
        len(sequences),
        profile_length
    )


# ============================================================
# 6. Helpers for evolving sequences toward a target profile
# ============================================================

def single_mutants(sequence, mutable_start, mutable_end):
    """
    Generate all possible single-base substitutions within
    sequence[mutable_start:mutable_end].

    The end coordinate is exclusive.
    """

    mutants = []

    sequence_list = list(sequence)

    for position in range(
        mutable_start,
        mutable_end
    ):

        original_base = sequence_list[position]

        for alternative_base in BASES:

            if alternative_base == original_base:
                continue

            mutant = sequence_list.copy()

            mutant[position] = alternative_base

            mutants.append(
                "".join(mutant)
            )

    return mutants


def rms_to_target(profiles, target):
    """RMS difference between profiles and one target profile."""

    return np.sqrt(
        np.mean(
            (profiles - target) ** 2,
            axis=1
        )
    )


def evolve_sequence_to_profile(
    target_profile,
    sequence_length,
    rounds=15,
    fixed_adapters=False
):
    """
    Generate one random sequence and iteratively optimize it
    toward a specified intrinsic-cyclisability profile.

    During each round all possible allowed single-base
    substitutions are evaluated.

    The parent sequence is also included as a candidate,
    allowing a round to make no change if no mutation
    improves the match.
    """

    target_profile = np.asarray(
        target_profile,
        dtype=float
    )

    expected_profile_length = (
        sequence_length - WINDOW + 1
    )

    assert len(target_profile) == expected_profile_length


    # --------------------------------------------------------
    # Starting sequence and mutable coordinates
    # --------------------------------------------------------

    if fixed_adapters:

        assert sequence_length == LIBRARY_LENGTH

        current_sequence = make_library_sequence(
            random_dna(VARIABLE_LENGTH)
        )

        mutable_start = len(LEFT_ADAPTER)

        mutable_end = (
            mutable_start
            + VARIABLE_LENGTH
        )

    else:

        current_sequence = random_dna(
            sequence_length
        )

        mutable_start = 0
        mutable_end = sequence_length


    # --------------------------------------------------------
    # Iterative optimization
    # --------------------------------------------------------

    for round_number in range(
        1,
        rounds + 1
    ):

        candidates = [
            current_sequence
        ]

        candidates.extend(
            single_mutants(
                current_sequence,
                mutable_start,
                mutable_end
            )
        )

        candidate_profiles = predict_profiles(
            candidates
        )

        rms = rms_to_target(
            candidate_profiles,
            target_profile
        )

        best_index = int(
            np.argmin(rms)
        )

        current_sequence = candidates[
            best_index
        ]

    return current_sequence


def evolve_multiple_sequences_to_targets(
    targets,
    sequence_length,
    fixed_adapters=False,
    rounds=15
):
    """
    Generate one optimized sequence for each target profile.
    """

    output = []

    for i, target in enumerate(targets):

        print(
            f"Optimizing target {i + 1}/{len(targets)}"
        )

        sequence = evolve_sequence_to_profile(
            target_profile=target,
            sequence_length=sequence_length,
            rounds=rounds,
            fixed_adapters=fixed_adapters
        )

        output.append(sequence)

    return output


# ============================================================
# 7. s1
#    100 native yeast nucleosomal coding sequences
# ============================================================

native_500_to_599 = (
    native_coding_sequences[500:600]
)


assert len(native_500_to_599) == 100


s1 = [
    coding_to_library_sequence(sequence)
    for sequence in native_500_to_599
]


assert len(s1) == 100


# ============================================================
# 8. s2
#    100 intrinsic-cyclisability mimics of s1
# ============================================================

# The targets are the complete 248-point profiles of the
# original 297-bp native coding sequences.

native_500_to_599_profiles = predict_profiles(
    native_500_to_599
)


assert native_500_to_599_profiles.shape == (
    100,
    248
)


# Generate 100 unrelated 297-bp sequences whose predicted
# profiles approximate the corresponding native profiles.

s2_full_length = (
    evolve_multiple_sequences_to_targets(
        targets=native_500_to_599_profiles,
        sequence_length=297,
        fixed_adapters=False,
        rounds=15
    )
)


# Only the central 113 bp of each optimized sequence enters
# the experimental library.

s2 = [
    coding_to_library_sequence(sequence)
    for sequence in s2_full_length
]


assert len(s2) == 100


# ============================================================
# 9. s3
#    20 sequences with a prescribed Gaussian profile
# ============================================================

GAUSSIAN_PEAK = 0.0
GAUSSIAN_BASELINE = -0.15
GAUSSIAN_SD = 20


x = np.arange(
    LIBRARY_PROFILE_LENGTH
)

center = (
    LIBRARY_PROFILE_LENGTH - 1
) / 2


gaussian_target = (

    GAUSSIAN_BASELINE

    +

    (
        GAUSSIAN_PEAK
        - GAUSSIAN_BASELINE
    )

    * np.exp(
        -0.5
        * (
            (x - center)
            / GAUSSIAN_SD
        ) ** 2
    )
)


s3 = evolve_multiple_sequences_to_targets(
    targets=[
        gaussian_target
        for _ in range(20)
    ],
    sequence_length=147,
    fixed_adapters=True,
    rounds=15
)


assert len(s3) == 20


# ============================================================
# 10. s4
#     20 sequences matching native nucleosome #510
# ============================================================

native_510 = (
    native_coding_sequences[509]
)


native_510_library = (
    coding_to_library_sequence(
        native_510
    )
)


native_510_target = (
    predict_profiles(
        [native_510_library]
    )[0]
)


s4 = evolve_multiple_sequences_to_targets(
    targets=[
        native_510_target
        for _ in range(20)
    ],
    sequence_length=147,
    fixed_adapters=True,
    rounds=15
)


assert len(s4) == 20


# ============================================================
# 11. s5
#     20 sequences matching native nucleosome #545
# ============================================================

native_545 = (
    native_coding_sequences[544]
)


native_545_library = (
    coding_to_library_sequence(
        native_545
    )
)


native_545_target = (
    predict_profiles(
        [native_545_library]
    )[0]
)


s5 = evolve_multiple_sequences_to_targets(
    targets=[
        native_545_target
        for _ in range(20)
    ],
    sequence_length=147,
    fixed_adapters=True,
    rounds=15
)


assert len(s5) == 20


# ============================================================
# 12. Standard genetic code
# ============================================================

CODON_TABLE = {

    "A": ["GCT", "GCC", "GCA", "GCG"],
    "C": ["TGT", "TGC"],
    "D": ["GAT", "GAC"],
    "E": ["GAA", "GAG"],
    "F": ["TTT", "TTC"],

    "G": ["GGT", "GGC", "GGA", "GGG"],
    "H": ["CAT", "CAC"],
    "I": ["ATT", "ATC", "ATA"],
    "K": ["AAA", "AAG"],

    "L": [
        "TTA", "TTG",
        "CTT", "CTC", "CTA", "CTG"
    ],

    "M": ["ATG"],
    "N": ["AAT", "AAC"],

    "P": [
        "CCT", "CCC", "CCA", "CCG"
    ],

    "Q": ["CAA", "CAG"],

    "R": [
        "CGT", "CGC", "CGA",
        "CGG", "AGA", "AGG"
    ],

    "S": [
        "TCT", "TCC", "TCA",
        "TCG", "AGT", "AGC"
    ],

    "T": [
        "ACT", "ACC", "ACA", "ACG"
    ],

    "V": [
        "GTT", "GTC", "GTA", "GTG"
    ],

    "W": ["TGG"],
    "Y": ["TAT", "TAC"]
}


# ============================================================
# 13. Generate synonymous variants of native sequence #345
# ============================================================

native_345_protein = (
    native_coding_proteins[345]
)


assert len(native_345_protein) == 99


def random_reverse_translation(
    protein_sequence
):
    """
    Reverse translate a protein by choosing independently
    and uniformly among synonymous codons.
    """

    codons = [

        random.choice(
            CODON_TABLE[amino_acid]
        )

        for amino_acid
        in protein_sequence
    ]

    return "".join(codons)


# Generate 1,000 synonymous 297-bp DNA sequences.

codon_variants_297 = [

    random_reverse_translation(
        native_345_protein
    )

    for _ in range(1000)
]


assert all(
    len(sequence) == 297
    for sequence in codon_variants_297
)


# Convert to the 147-bp experimental format.

codon_variants_147 = [

    coding_to_library_sequence(sequence)

    for sequence
    in codon_variants_297
]


# ============================================================
# 14. Rank codon variants by maximum cyclisability
# ============================================================

codon_profiles = predict_profiles(
    codon_variants_147
)


maximum_cyclisability = (
    codon_profiles.max(axis=1)
)


ranked_indices = np.argsort(
    maximum_cyclisability
)


# ============================================================
# 15. s6
#     Select 110 codon-randomized variants
# ============================================================

s6 = []


# Take the first 10 sequences from each successive
# 100-sequence block of the cyclisability ranking.

for start in range(
    0,
    1000,
    100
):

    block = ranked_indices[
        start:start + 10
    ]

    s6.extend(
        codon_variants_147[i]
        for i in block
    )


# Add the 10 sequences with the highest maximum
# intrinsic cyclisability.

s6.extend(

    codon_variants_147[i]

    for i in ranked_indices[-10:]
)


assert len(s6) == 110


# ============================================================
# 16. s7
#     Original native sequence #345
# ============================================================

native_345 = (
    native_coding_sequences[345]
)


s7 = [
    coding_to_library_sequence(
        native_345
    )
]


assert len(s7) == 1


# ============================================================
# 17. s8
#     828 fully random controls
# ============================================================

s8 = [

    make_library_sequence(
        random_dna(
            VARIABLE_LENGTH
        )
    )

    for _ in range(828)
]


assert len(s8) == 828


# ============================================================
# 18. Assemble sequences 802-2000
# ============================================================

sequences_802_to_2000 = (

    s1
    + s2
    + s3
    + s4
    + s5
    + s6
    + s7
    + s8
)


assert len(
    sequences_802_to_2000
) == 1199


assert all(
    len(sequence) == 147
    for sequence in sequences_802_to_2000
)


# ============================================================
# 19. Construct matching labels
# ============================================================

labels_802_to_2000 = (

    ["Yeast"] * 100

    + ["Mimic yeast (7)"] * 100

    + ["Gaussian flex"] * 20

    + ["Native 510 mimic"] * 20

    + ["Native 545 mimic"] * 20

    + ["Codon-randomized 345"] * 110

    + ["Native 345"] * 1

    + ["Random"] * 828
)


assert len(
    labels_802_to_2000
) == 1199


# ============================================================
# 20. Make an annotated table
# ============================================================

library_802_to_2000 = pd.DataFrame({

    "Library_position":
        np.arange(
            802,
            2001
        ),

    "Sequence":
        sequences_802_to_2000,

    "Label":
        labels_802_to_2000
})


# ============================================================
# 21. Final checks
# ============================================================

expected_counts = {

    "Yeast": 100,

    "Mimic yeast (7)": 100,

    "Gaussian flex": 20,

    "Native 510 mimic": 20,

    "Native 545 mimic": 20,

    "Codon-randomized 345": 110,

    "Native 345": 1,

    "Random": 828
}


observed_counts = (
    library_802_to_2000[
        "Label"
    ]
    .value_counts()
    .to_dict()
)


assert observed_counts == expected_counts


print(
    "\nSequences 802-2000:"
)

print(
    library_802_to_2000[
        "Label"
    ]
    .value_counts(
        sort=False
    )
)


print(
    "\nTotal:",
    len(
        library_802_to_2000
    )
)


# ============================================================
# OUTPUTS
# ============================================================

# Individual sequence sets:
#
#     s1
#         100 native yeast sequences
#
#     s2
#         100 intrinsic-cyclisability mimics of s1
#
#     s3
#         20 Gaussian-profile sequences
#
#     s4
#         20 native-510-profile mimics
#
#     s5
#         20 native-545-profile mimics
#
#     s6
#         110 synonymous codon-randomized variants
#         of native sequence #345
#
#     s7
#         original native sequence #345
#
#     s8
#         828 random controls
#
#
# Complete ordered sequence list:
#
#     sequences_802_to_2000
#
#
# Complete annotated table:
#
#     library_802_to_2000
#
#
# The latter contains:
#
#     Library_position
#     Sequence
#     Label