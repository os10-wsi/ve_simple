#!/bin/bash

#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -n 12
#BSUB -q "long"
#BSUB -R "select[mem>3000] rusage[mem=3000] span[hosts=1]"
#BSUB -M 3000
#BSUB -G team354

module load ISG/IRODS/1.0
module load HGI/softpack/users/tw10/Bisulfite/1

samtools collate -O -u -@ 12 irods://seq/illumina/runs/53/53345/plex29/53345#29.cram pfx-OTAR309717025663 | samtools fastq -F 0x900 -@ 12 -0 OTAR309717025663.fastq.gz -N -1 OTAR309717025663_1.fastq.gz -2 OTAR309717025663_2.fastq.gz -
