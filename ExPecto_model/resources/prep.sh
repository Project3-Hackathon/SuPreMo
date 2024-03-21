#!/bin/bash


python build_hg38.py --gtf gencode.v45.basic.annotation.gtf --gene_list geneanno.csv --output geneanno.hg38.bed

# then build tabix index
