
"""
SCRIPT NAME:
Code 42 Figure 4-4 positional synonymous codon usage.py


WHAT THIS SCRIPT DOES:

Reconstructs Figure 4.4 from the original genomic inputs.

The script:

1. Loads the SacCer3 yeast genome.

2. Loads the SacCer3 native nucleosome dyad positions.

3. Loads the Pugh gene annotation.

4. Finds native nucleosome dyads lying within each annotated gene.

5. Around each gene-body nucleosome, constructs a 297-bp DNA
   sequence corresponding to exactly 99 codons.

6. Shifts the nominal dyad-centred 297-bp window upstream by
   0, 1, or 2 bp where necessary so that the extracted sequence
   is in the correct coding frame.

7. Reverse-complements sequences from minus-strand genes so all
   sequences are represented in transcriptional orientation.

8. Removes sequences containing ambiguous bases or internal
   stop codons.

9. Counts every non-stop codon at each of the 99 codon positions.

10. Converts those counts to synonymous-codon usage frequencies.

For each amino acid and each position:

    sum of frequencies of its synonymous codons = 1


THESIS SECTION / FIGURE:

Figure 4.4


INPUT FILES:

sacCer3.fa

    SacCer3 genome sequence.


NucleosomeSacCer3.txt

    SacCer3 native nucleosome positions produced by the
    SacCer2 -> SacCer3 liftOver.

    Expected columns:

        chromosome
        dyad


Pugh.txt

    Gene annotation.

    Required columns:

        Chrom
        Strand
        Experiment_Left
        Experiment_Right
        Systematic ID
        Feature class Level 1
        SGD_Left
        SGD_Right


VARIABLES THAT MUST ALREADY EXIST:

None.


OUTPUT VARIABLES:

nuc_gene_seq

    List of valid 297-bp coding nucleosome sequences.


df_coding_nucleosomes

    Metadata for each retained 297-bp sequence.


codon_usage_counts_df

    Raw counts for all 61 non-stop codons at positions 1-99.


codon_usage_df

    Synonymous-codon usage frequencies.


figure_4_4

    Long-format data for Figure 4.4.


NOTES:

The historical code produced:

    40,187

297-bp sequences after removal of sequences containing internal
stop codons.

This number is retained as a useful reconstruction check rather
than enforced as an assertion.
"""


import numpy as np
import pandas as pd
from collections import defaultdict


# ================================================================
# 1. Input files
# ================================================================

GENOME_FILE = "sacCer3.fa"

NUCLEOSOME_FILE = "NucleosomeSacCer3.txt"

PUGH_FILE = "Pugh.txt"


# ================================================================
# 2. Read SacCer3 genome
# ================================================================

chromosome_sequences_roman = {}


with open(GENOME_FILE, "r") as f:

    current_chrom = None
    current_sequence = []


    for line in f:

        line = line.strip()


        if line.startswith(">"):

            if current_chrom is not None:

                chromosome_sequences_roman[
                    current_chrom
                ] = "".join(
                    current_sequence
                ).upper()


            current_chrom = (
                line[1:]
                .split()[0]
            )

            current_sequence = []


        else:

            current_sequence.append(
                line
            )


    if current_chrom is not None:

        chromosome_sequences_roman[
            current_chrom
        ] = "".join(
            current_sequence
        ).upper()


# ================================================================
# 3. Convert chromosome names to chr1 ... chr16
# ================================================================

roman_to_arabic = {

    "chrI": "chr1",
    "chrII": "chr2",
    "chrIII": "chr3",
    "chrIV": "chr4",
    "chrV": "chr5",
    "chrVI": "chr6",
    "chrVII": "chr7",
    "chrVIII": "chr8",
    "chrIX": "chr9",
    "chrX": "chr10",
    "chrXI": "chr11",
    "chrXII": "chr12",
    "chrXIII": "chr13",
    "chrXIV": "chr14",
    "chrXV": "chr15",
    "chrXVI": "chr16"
}


chromosome_sequences = {}


for chrom_roman, chrom_arabic in (
    roman_to_arabic.items()
):

    if chrom_roman in chromosome_sequences_roman:

        chromosome_sequences[
            chrom_arabic
        ] = chromosome_sequences_roman[
            chrom_roman
        ]


print(
    "Loaded chromosomes:",
    len(chromosome_sequences)
)


for chrom in [
    f"chr{i}"
    for i in range(1, 17)
]:

    if chrom not in chromosome_sequences:

        raise ValueError(
            f"{chrom} was not found in {GENOME_FILE}"
        )


# ================================================================
# 4. Load SacCer3 native nucleosome dyads
# ================================================================

df3 = pd.read_csv(

    NUCLEOSOME_FILE,

    sep="\t",

    header=None,

    names=[
        "chromosome",
        "dyad"
    ]
)


df3["chromosome"] = (
    df3["chromosome"]
    .replace(
        roman_to_arabic
    )
)


df3["dyad"] = pd.to_numeric(
    df3["dyad"],
    errors="coerce"
)


df3 = df3[
    df3["dyad"].notna()
].copy()


df3["dyad"] = (
    df3["dyad"]
    .astype(int)
)


print(
    "Native nucleosome dyads:",
    len(df3)
)


# ================================================================
# 5. Store dyads chromosome by chromosome
# ================================================================

nucleosome_dict = {

    chrom:
        sorted(
            group[
                "dyad"
            ].tolist()
        )

    for chrom, group
    in df3.groupby(
        "chromosome"
    )
}


# ================================================================
# 6. Load Pugh gene annotation
# ================================================================

df4 = pd.read_csv(
    PUGH_FILE,
    sep="\t"
)


required_columns = [

    "Chrom",
    "Strand",

    "Experiment_Left",
    "Experiment_Right",

    "Systematic ID",
    "Feature class Level 1",

    "SGD_Left",
    "SGD_Right"
]


missing_columns = [

    column

    for column in required_columns

    if column not in df4.columns
]


if missing_columns:

    raise ValueError(
        "Missing Pugh columns: "
        +
        ", ".join(
            missing_columns
        )
    )


df6 = df4[
    required_columns
].copy()


# ================================================================
# 7. Convert coordinate columns to numeric
# ================================================================

coordinate_columns = [

    "Experiment_Left",
    "Experiment_Right",

    "SGD_Left",
    "SGD_Right"
]


for column in coordinate_columns:

    df6[column] = pd.to_numeric(

        df6[column],

        errors="coerce"

    ).astype(
        "Int64"
    )


# ================================================================
# 8. Find native nucleosome centres inside each gene
# ================================================================

def get_nucleosome_centers(row):

    chrom = row[
        "Chrom"
    ]

    strand = row[
        "Strand"
    ]

    left = row[
        "Experiment_Left"
    ]

    right = row[
        "Experiment_Right"
    ]


    if (
        pd.isna(left)
        or
        pd.isna(right)
    ):

        return []


    if chrom not in nucleosome_dict:

        return []


    centers = [

        dyad

        for dyad
        in nucleosome_dict[
            chrom
        ]

        if (
            left
            <=
            dyad
            <=
            right
        )
    ]


    # Put nucleosomes into transcriptional order.

    if strand == "-":

        centers = sorted(
            centers,
            reverse=True
        )

    else:

        centers = sorted(
            centers
        )


    return centers


df6[
    "nucleosome centers"
] = df6.apply(

    get_nucleosome_centers,

    axis=1
)


# ================================================================
# 9. Reverse-complement function
# ================================================================

def reverse_complement(sequence):

    complement = str.maketrans(

        "ATGCatgc",

        "TACGtacg"
    )


    return (
        sequence
        .translate(
            complement
        )[::-1]
    )


# ================================================================
# 10. Translation function
# ================================================================

genetic_code = {

    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",

    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",

    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",

    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",

    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",

    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",

    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",

    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",

    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",

    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",

    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",

    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",

    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",

    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",

    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",

    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G"
}


def translate_frame1(sequence):

    amino_acids = []


    for i in range(
        0,
        len(sequence),
        3
    ):

        codon = sequence[
            i:i + 3
        ]


        if len(codon) != 3:

            continue


        amino_acids.append(

            genetic_code.get(
                codon,
                "X"
            )
        )


    return "".join(
        amino_acids
    )


# ================================================================
# 11. Construct the 297-bp coding nucleosome sequences
# ================================================================
#
# This follows the construction in code_history.py.
#
# A nominal 297-bp window is:
#
#     dyad - 148  ...  dyad + 148
#
# inclusive.
#
# The window can then be shifted upstream by 0, 1, or 2 bp to
# make its first base correspond to the beginning of a codon.
#
# For minus-strand genes, "upstream" is toward increasing genomic
# coordinates.
# ================================================================

coding_nucleosome_rows = []


for gene_index, row in (
    df6.iterrows()
):

    chrom = row[
        "Chrom"
    ]

    strand = row[
        "Strand"
    ]


    # ------------------------------------------------------------
    # Required chromosome
    # ------------------------------------------------------------

    if chrom not in chromosome_sequences:

        continue


    genome = chromosome_sequences[
        chrom
    ]


    # ------------------------------------------------------------
    # Need CDS coordinates
    # ------------------------------------------------------------

    if (
        pd.isna(
            row[
                "SGD_Left"
            ]
        )
        or
        pd.isna(
            row[
                "SGD_Right"
            ]
        )
    ):

        continue


    centers = row[
        "nucleosome centers"
    ]


    if len(centers) == 0:

        continue


    # ============================================================
    # 12. Determine translation start and end
    # ============================================================

    if strand == "+":

        CDS_start = int(
            row[
                "SGD_Left"
            ]
        )

        CDS_end = int(
            row[
                "SGD_Right"
            ]
        )


    elif strand == "-":

        CDS_start = int(
            row[
                "SGD_Right"
            ]
        )

        CDS_end = int(
            row[
                "SGD_Left"
            ]
        )


    else:

        continue


    lower = min(
        CDS_start,
        CDS_end
    )

    upper = max(
        CDS_start,
        CDS_end
    )


    # ============================================================
    # 13. Process every nucleosome within this gene
    # ============================================================

    for nucleosome_index, dyad in enumerate(
        centers
    ):

        # --------------------------------------------------------
        # Nominal 297-bp window centred on the dyad.
        #
        # These are 1-based inclusive genomic coordinates.
        # --------------------------------------------------------

        start = (
            dyad
            -
            148
        )

        end = (
            dyad
            +
            148
        )


        valid = False

        chosen_shift = None

        adj_start = None

        adj_end = None


        # --------------------------------------------------------
        # Try shifts of 0, 1 and 2 bp.
        # --------------------------------------------------------

        for shift in range(
            3
        ):


            if strand == "+":

                # Upstream is toward decreasing coordinate.

                test_start = (
                    start
                    -
                    shift
                )

                test_end = (
                    end
                    -
                    shift
                )


                # Whole 297-bp fragment must remain within CDS.

                if (
                    test_start
                    <
                    lower
                    or
                    test_end
                    >
                    upper
                ):

                    continue


                # First base must be at the beginning of a codon.

                if (
                    (
                        test_start
                        -
                        CDS_start
                    )
                    %
                    3
                    ==
                    0
                ):

                    adj_start = (
                        test_start
                    )

                    adj_end = (
                        test_end
                    )

                    chosen_shift = (
                        shift
                    )

                    valid = True

                    break


            else:

                # For a minus-strand gene, upstream is toward
                # increasing genomic coordinate.

                test_start = (
                    start
                    +
                    shift
                )

                test_end = (
                    end
                    +
                    shift
                )


                if (
                    test_start
                    <
                    lower
                    or
                    test_end
                    >
                    upper
                ):

                    continue


                # After reverse complementation, the first base
                # of the sequence is genomic test_end.

                if (
                    (
                        CDS_start
                        -
                        test_end
                    )
                    %
                    3
                    ==
                    0
                ):

                    adj_start = (
                        test_start
                    )

                    adj_end = (
                        test_end
                    )

                    chosen_shift = (
                        shift
                    )

                    valid = True

                    break


        if not valid:

            continue


        # ========================================================
        # 14. Extract genomic sequence
        # ========================================================
        #
        # Genomic coordinates above are 1-based inclusive.
        #
        # Python:
        #
        #     genome[start - 1 : end]
        #
        # therefore returns exactly:
        #
        #     end - start + 1 = 297 bp
        # ========================================================

        sequence = genome[
            adj_start - 1
            :
            adj_end
        ]


        if len(
            sequence
        ) != 297:

            continue


        sequence = sequence.upper()


        if "N" in sequence:

            continue


        # --------------------------------------------------------
        # Put minus-strand genes into transcriptional orientation.
        # --------------------------------------------------------

        if strand == "-":

            sequence = (
                reverse_complement(
                    sequence
                )
            )


        # --------------------------------------------------------
        # Translate.
        # --------------------------------------------------------

        protein = (
            translate_frame1(
                sequence
            )
        )


        if len(
            protein
        ) != 99:

            continue


        # --------------------------------------------------------
        # Historical filtering:
        #
        # remove sequences containing internal stop codons.
        #
        # The final codon is allowed to be a stop in the historical
        # expression:
        #
        #     "*" not in translate_frame1(seq)[:-1]
        # --------------------------------------------------------

        if "*" in protein[:-1]:

            continue


        coding_nucleosome_rows.append(
            {

                "Gene_index":
                    gene_index,

                "Systematic_ID":
                    row[
                        "Systematic ID"
                    ],

                "Chromosome":
                    chrom,

                "Strand":
                    strand,

                "Nucleosome_number":
                    nucleosome_index
                    +
                    1,

                "Dyad":
                    dyad,

                "Frame_shift_bp":
                    chosen_shift,

                "Genomic_start":
                    adj_start,

                "Genomic_end":
                    adj_end,

                "297_bp_coding_sequence":
                    sequence,

                "99_aa_sequence":
                    protein
            }
        )


# ================================================================
# 15. Store constructed sequences
# ================================================================

df_coding_nucleosomes = pd.DataFrame(
    coding_nucleosome_rows
)


nuc_gene_seq = (
    df_coding_nucleosomes[
        "297_bp_coding_sequence"
    ]
    .tolist()
)


print()
print(
    "297-bp coding nucleosome sequences:",
    len(
        nuc_gene_seq
    )
)


print(
    "Historical code reported: 40,187"
)


print(
    "Sequence lengths:",
    set(
        map(
            len,
            nuc_gene_seq
        )
    )
)


# ================================================================
# 16. Construction checks
# ================================================================

if len(
    nuc_gene_seq
) == 0:

    raise ValueError(
        "No valid 297-bp coding nucleosome sequences were found."
    )


if set(
    map(
        len,
        nuc_gene_seq
    )
) != {297}:

    raise ValueError(
        "Not all retained sequences are 297 bp."
    )


# ================================================================
# 17. Non-stop genetic code for Figure 4.4
# ================================================================

genetic_code_no_stop = {

    codon:
        amino_acid

    for codon, amino_acid
    in genetic_code.items()

    if amino_acid
    !=
    "*"
}


assert len(
    genetic_code_no_stop
) == 61


# ================================================================
# 18. Group synonymous codons by amino acid
# ================================================================

aa_to_codons = defaultdict(
    list
)


for codon, amino_acid in (
    genetic_code_no_stop.items()
):

    aa_to_codons[
        amino_acid
    ].append(
        codon
    )


aa_to_codons = {

    amino_acid:
        sorted(
            codons
        )

    for amino_acid, codons
    in aa_to_codons.items()
}


# ================================================================
# 19. Count every codon at every position
# ================================================================
#
# counts[amino_acid][codon]
#
# gives an array of length 99.
# ================================================================

counts = {

    amino_acid: {

        codon:
            np.zeros(
                99,
                dtype=int
            )

        for codon in codons
    }

    for amino_acid, codons
    in aa_to_codons.items()
}


for sequence in nuc_gene_seq:


    codons = [

        sequence[
            i:i + 3
        ]

        for i in range(
            0,
            297,
            3
        )
    ]


    assert len(
        codons
    ) == 99


    for position, codon in enumerate(
        codons
    ):


        # A terminal stop codon can theoretically remain because
        # the historical filtering excluded only internal stops.
        #
        # Stop codons are not part of Figure 4.4.

        if codon not in genetic_code_no_stop:

            continue


        amino_acid = (
            genetic_code_no_stop[
                codon
            ]
        )


        counts[
            amino_acid
        ][
            codon
        ][
            position
        ] += 1


# ================================================================
# 20. Convert counts to synonymous usage frequencies
# ================================================================
#
# For each amino acid separately:
#
#                    count of codon c at position p
# frequency(c,p) = ----------------------------------
#                   count of all synonymous codons
#                         at position p
#
# Therefore the synonymous codon frequencies for an amino acid
# sum to 1 at every position where that amino acid occurs.
# ================================================================

frequencies = {}


for amino_acid, codon_dict in (
    counts.items()
):


    total = np.zeros(
        99,
        dtype=float
    )


    for codon_counts in (
        codon_dict.values()
    ):

        total += (
            codon_counts
        )


    frequencies[
        amino_acid
    ] = {}


    for codon, codon_counts in (
        codon_dict.items()
    ):


        frequency = np.full(
            99,
            np.nan,
            dtype=float
        )


        valid = (
            total
            >
            0
        )


        frequency[
            valid
        ] = (

            codon_counts[
                valid
            ]

            /

            total[
                valid
            ]
        )


        frequencies[
            amino_acid
        ][
            codon
        ] = (
            frequency
        )


# ================================================================
# 21. Verify synonymous frequencies
# ================================================================

for amino_acid, codons in (
    aa_to_codons.items()
):


    frequency_matrix = np.vstack(
        [

            frequencies[
                amino_acid
            ][
                codon
            ]

            for codon in codons
        ]
    )


    count_matrix = np.vstack(
        [

            counts[
                amino_acid
            ][
                codon
            ]

            for codon in codons
        ]
    )


    total_counts = np.sum(
        count_matrix,
        axis=0
    )


    valid = (
        total_counts
        >
        0
    )


    summed_frequency = np.nansum(
        frequency_matrix,
        axis=0
    )


    if not np.allclose(

        summed_frequency[
            valid
        ],

        1.0

    ):

        raise ValueError(
            "Synonymous frequencies do not sum to 1 "
            f"for amino acid {amino_acid}."
        )


# ================================================================
# 22. Wide raw-count dataframe
# ================================================================

count_rows = []


for amino_acid in sorted(
    aa_to_codons
):


    for codon in (
        aa_to_codons[
            amino_acid
        ]
    ):


        row = {

            "Amino_Acid":
                amino_acid,

            "Codon":
                codon
        }


        for position in range(
            99
        ):

            row[
                f"Pos{position + 1}"
            ] = (

                counts[
                    amino_acid
                ][
                    codon
                ][
                    position
                ]
            )


        count_rows.append(
            row
        )


codon_usage_counts_df = pd.DataFrame(
    count_rows
)


# ================================================================
# 23. Wide frequency dataframe
# ================================================================

frequency_rows = []


for amino_acid in sorted(
    aa_to_codons
):


    for codon in (
        aa_to_codons[
            amino_acid
        ]
    ):


        row = {

            "Amino_Acid":
                amino_acid,

            "Codon":
                codon
        }


        for position in range(
            99
        ):

            row[
                f"Pos{position + 1}"
            ] = (

                frequencies[
                    amino_acid
                ][
                    codon
                ][
                    position
                ]
            )


        frequency_rows.append(
            row
        )


codon_usage_df = pd.DataFrame(
    frequency_rows
)


# ================================================================
# 24. Convert codon positions to DNA coordinates
# ================================================================
#
# Historical description:
#
#     region number was multiplied by 3 and offset so the
#     nucleosome centre was zero.
#
# With 99 codon positions:
#
#     codon 1  = -147 bp
#     codon 50 =    0 bp
#     codon 99 = +147 bp
# ================================================================

codon_positions = np.arange(
    1,
    100
)


position_bp = (

    codon_positions
    -
    50

) * 3


assert (
    position_bp[0]
    ==
    -147
)


assert (
    position_bp[49]
    ==
    0
)


assert (
    position_bp[-1]
    ==
    147
)


# ================================================================
# 25. Long-format Figure 4.4 data
# ================================================================

figure_rows = []


for amino_acid in sorted(
    aa_to_codons
):


    for codon in (
        aa_to_codons[
            amino_acid
        ]
    ):


        for position_index in range(
            99
        ):


            figure_rows.append(
                {

                    "Amino_Acid":
                        amino_acid,

                    "Codon":
                        codon,

                    "Codon_position":
                        position_index
                        +
                        1,

                    "Position_bp":
                        position_bp[
                            position_index
                        ],

                    "Count":
                        counts[
                            amino_acid
                        ][
                            codon
                        ][
                            position_index
                        ],

                    "Usage_frequency":
                        frequencies[
                            amino_acid
                        ][
                            codon
                        ][
                            position_index
                        ]
                }
            )


figure_4_4 = pd.DataFrame(
    figure_rows
)


# ================================================================
# 26. Final checks
# ================================================================

assert len(
    codon_usage_df
) == 61


assert len(
    figure_4_4
) == (
    61
    *
    99
)


print()
print(
    "Figure 4.4 data ready."
)


print(
    "Retained coding nucleosome sequences:",
    len(
        nuc_gene_seq
    )
)


print(
    "Number of amino acids:",
    len(
        aa_to_codons
    )
)


print(
    "Number of non-stop codons:",
    len(
        codon_usage_df
    )
)


print(
    "Positions per codon:",
    99
)


print(
    "DNA-coordinate range:",
    position_bp[0],
    "to",
    position_bp[-1],
    "bp"
)


print()
print(
    figure_4_4.head()
)