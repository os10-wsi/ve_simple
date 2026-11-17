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

samtools collate -O -u -@ 12 irods://seq/illumina/runs/53/53345/plex25/53345#25.cram pfx-OTAR309717025659 | samtools fastq -F 0x900 -@ 12 -0 OTAR309717025659.fastq.gz -N -1 OTAR309717025659_1.fastq.gz -2 OTAR309717025659_2.fastq.gz -
