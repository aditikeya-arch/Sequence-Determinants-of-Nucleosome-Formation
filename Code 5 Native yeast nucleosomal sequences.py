

"""
Extract native yeast nucleosomal sequences and NCP scores
=========================================================

PURPOSE
-------
Generate the native yeast nucleosome dataset used for the analyses
associated with Figure 2.11.

The starting dataset contains experimentally mapped nucleosome dyad
positions in the SacCer2 genome assembly together with an NCP score.

Because the chromosome sequences used for subsequent sequence analysis
are from the SacCer3 assembly, nucleosome coordinates are first converted
from SacCer2 to SacCer3 using UCSC liftOver.

The final output contains, for every successfully mapped nucleosome:

    chromosome
    SacCer3 dyad coordinate
    NCP score
    147-bp DNA sequence centred on the dyad


DATA SOURCES
------------
1. Nucleosome positions and NCP scores

   Kaplan et al. (2012)
   "The DNA-encoded nucleosome organization of a eukaryotic genome"
   Nature 458? [use full thesis/reference-list citation when depositing]

   Article / associated data:
   https://www.nature.com/articles/nature11142

   The downloaded table is saved locally as:

       NucleosomeSacCer2.txt

   The relevant columns are:

       chromosome
       dyad coordinate
       NCP score

   Coordinates in this file are SacCer2 coordinates.


2. SacCer2 -> SacCer3 coordinate conversion

   UCSC Genome Browser liftOver is used to convert the published
   SacCer2 coordinates to SacCer3.

   UCSC liftOver:
   https://genome.ucsc.edu/cgi-bin/hgLiftOver

   The corresponding chain file is:

       sacCer2ToSacCer3.over.chain.gz

   available from the UCSC SacCer2 liftOver directory:

   https://hgdownload.soe.ucsc.edu/goldenPath/sacCer2/liftOver/


3. SacCer3 chromosome sequences

   SacCer3 is the April 2011 S. cerevisiae S288C assembly.

   Chromosome FASTA files can be obtained from the UCSC Genome Browser:

   https://hgdownload.cse.ucsc.edu/goldenPath/sacCer3/chromosomes/

   These chromosome sequences are assumed here to have already been
   loaded into:

       sequences_data

   where:

       sequences_data[0]  = chrI
       sequences_data[1]  = chrII
       ...
       sequences_data[15] = chrXVI


REQUIRED VARIABLES
------------------
sequences_data : list of str

    List of the 16 SacCer3 chromosome sequences, ordered chromosome
    I through chromosome XVI.


WORKFLOW
--------
NucleosomeSacCer2.txt
        |
        | create +/- 10 bp intervals around dyads
        v
SacCer2bed.bed
        |
        | UCSC liftOver: SacCer2 -> SacCer3
        v
SacCer3bed.bed
        |
        | recover SacCer3 dyad
        | extract 147-bp sequence
        v
NativeYeastNucleosomes.csv


IMPORTANT
---------
The liftOver step is external to Python and must be performed between
PART 1 and PART 2 of this script.
"""


import pandas as pd


# ============================================================
# PART 1
# Prepare SacCer2 coordinates for liftOver
# ============================================================


# ------------------------------------------------------------
# 1. Read published nucleosome coordinates
# ------------------------------------------------------------

df_saccer2 = pd.read_csv(
    "NucleosomeSacCer2.txt",
    sep=r"\s+",
    header=None,
    names=[
        "Chromosome",
        "Dyad",
        "NCP_score",
        "Unused"
    ]
)


# Only the first three columns are required.

df_saccer2 = df_saccer2[
    [
        "Chromosome",
        "Dyad",
        "NCP_score"
    ]
].copy()


# ------------------------------------------------------------
# 2. Give every nucleosome a unique ID
#
# This allows the NCP score to be associated with the correct
# nucleosome after liftOver.
# ------------------------------------------------------------

df_saccer2["Nucleosome_ID"] = [
    f"nuc_{i}"
    for i in range(len(df_saccer2))
]


# ------------------------------------------------------------
# 3. Construct a small interval around each SacCer2 dyad
#
# This reproduces the recovered analysis:
#
#     Start = Dyad - 10
#     End   = Dyad + 10
#
# These coordinates will be converted from SacCer2 to SacCer3.
# ------------------------------------------------------------

df_liftover = pd.DataFrame({

    "Chromosome":
        df_saccer2["Chromosome"],

    "Start":
        df_saccer2["Dyad"] - 10,

    "End":
        df_saccer2["Dyad"] + 10,

    "Nucleosome_ID":
        df_saccer2["Nucleosome_ID"]
})


# ------------------------------------------------------------
# 4. Write BED file
# ------------------------------------------------------------

df_liftover.to_csv(
    "SacCer2bed.bed",
    sep="\t",
    header=False,
    index=False
)


# ============================================================
# EXTERNAL STEP: UCSC liftOver
# ============================================================

#
# SacCer2bed.bed must now be converted from SacCer2 coordinates
# to SacCer3 coordinates.
#
# The required UCSC chain file is:
#
#     sacCer2ToSacCer3.over.chain.gz
#
# Example command-line usage:
#
# liftOver \
#     SacCer2bed.bed \
#     sacCer2ToSacCer3.over.chain.gz \
#     SacCer3bed.bed \
#     SacCer2_unmapped.bed
#
#
# INPUT:
#
#     SacCer2bed.bed
#
# OUTPUT:
#
#     SacCer3bed.bed
#
#     SacCer2_unmapped.bed
#
#
# The fourth BED column (Nucleosome_ID) should be retained
# during liftOver. This allows the lifted coordinates to be
# re-associated with the original NCP scores.
#
# After running liftOver, continue with PART 2 below.
#


# ============================================================
# PART 2
# Read lifted SacCer3 coordinates
# ============================================================


# ------------------------------------------------------------
# 5. Read lifted coordinates
# ------------------------------------------------------------

df_saccer3 = pd.read_csv(
    "SacCer3bed.bed",
    sep="\t",
    header=None,
    names=[
        "Chromosome",
        "Start",
        "End",
        "Nucleosome_ID"
    ]
)


# ------------------------------------------------------------
# 6. Recover the SacCer3 dyad
#
# The original interval was constructed as:
#
#     Start = Dyad - 10
#
# so the centre is recovered as:
#
#     Dyad = Start + 10
# ------------------------------------------------------------

df_saccer3["Dyad"] = (
    df_saccer3["Start"] + 10
)


# ------------------------------------------------------------
# 7. Reattach the published NCP score
# ------------------------------------------------------------

df_saccer3 = df_saccer3.merge(

    df_saccer2[
        [
            "Nucleosome_ID",
            "NCP_score"
        ]
    ],

    on="Nucleosome_ID",
    how="left"
)


# ============================================================
# 8. Map UCSC chromosome names onto sequences_data
# ============================================================

chromosome_to_index = {

    "chrI": 0,
    "chrII": 1,
    "chrIII": 2,
    "chrIV": 3,
    "chrV": 4,
    "chrVI": 5,
    "chrVII": 6,
    "chrVIII": 7,
    "chrIX": 8,
    "chrX": 9,
    "chrXI": 10,
    "chrXII": 11,
    "chrXIII": 12,
    "chrXIV": 13,
    "chrXV": 14,
    "chrXVI": 15
}


# ============================================================
# 9. Extract the 147-bp sequence around each dyad
# ============================================================

def get_nucleosome_sequence(chromosome, dyad):

    chromosome_index = chromosome_to_index[
        chromosome
    ]

    chromosome_sequence = sequences_data[
        chromosome_index
    ]

    dyad = int(dyad)

    # Extract 147 bp centred on the dyad:
    #
    # 73 bp upstream
    # + central base
    # + 73 bp downstream
    #
    # NOTE:
    # The exact +/-1 convention should be verified against the
    # coordinate convention used in the original analysis.

    sequence = chromosome_sequence[
        dyad - 73:
        dyad + 74
    ]

    return sequence


df_saccer3["Sequence"] = [

    get_nucleosome_sequence(
        chromosome,
        dyad
    )

    for chromosome, dyad
    in zip(
        df_saccer3["Chromosome"],
        df_saccer3["Dyad"]
    )
]


# ============================================================
# 10. Check sequence lengths
# ============================================================

df_saccer3["Sequence_length"] = (
    df_saccer3["Sequence"].str.len()
)


print(
    df_saccer3["Sequence_length"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 11. Retain complete 147-bp nucleosome sequences
# ============================================================

native_nucleosomes = (

    df_saccer3[
        df_saccer3["Sequence_length"] == 147
    ]

    [
        [
            "Chromosome",
            "Dyad",
            "NCP_score",
            "Sequence"
        ]
    ]

    .reset_index(drop=True)
)


print(
    "Number of native nucleosomes:",
    len(native_nucleosomes)
)


# ============================================================
# 12. Save final dataset
# ============================================================

native_nucleosomes.to_csv(
    "NativeYeastNucleosomes.csv",
    index=False
)