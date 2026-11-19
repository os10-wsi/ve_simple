# Datasets for scripts/new_datasets/*.sh. Edit, then run the stages in order from the repository root.
# Each dataset is NAME=ID_FILE; NAME becomes results_<NAME>/, data_<NAME>/ and the figure folders.
P=/lustre/scratch125/gengen/projects_v2/mp_stability/retrieve_proteomes

# Proteomes to download in stage 1 (reviewed UniProt). Mind the file quota (docs/new_datasets.md):
# 7227 fly, 6239 worm, 83333 E. coli, 284812 fission yeast, 7955 zebrafish, 44689 Dictyostelium,
# 10116 rat, 3702 Arabidopsis, 10090 mouse
TAXIDS="7227 6239 83333"

declare -A DATASETS=(
  [fly]=$P/uniprot_reviewed_7227.txt
  [worm]=$P/uniprot_reviewed_6239.txt
  [ecoli]=$P/uniprot_reviewed_83333.txt
  [families]=sets/families/all_family_ids.txt
)

# Membrane-family selection (stage 1)
FAMILY_N=500                  # representatives per family
FAMILY_THREADS=16
MMSEQS=${MMSEQS:-mmseqs}      # path to the MMseqs2 binary

# Datasets shown together on the talk structure figure (stage 5)
STRUCT_SETS=human,yeast,fly,worm,ecoli
