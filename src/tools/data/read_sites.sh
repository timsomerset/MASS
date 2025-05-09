#!/bin/bash

cwd=$(pwd)

bam_path=$1
out_dir=$2
sites_file=$3
ref_gnm=$4

# Check if bam path exists
if [[ ! -f "$bam_path" ]]; then
    echo "(read_sites.sh; Error) BAM file '$bam_path' does not exist."
    exit 1
fi

bam_path=$(realpath --relative-to="$cwd" "$bam_path" 2>/dev/null || echo "$bam_path")
bam_file="${bam_path##*/}"
bam_name="${bam_file%.*}"

out_file="$out_dir/$bam_name.tsv"

if [[ -f "$out_file" ]]; then 
    echo "(read_sites.sh; Job Stopped) Sites file already exists for $bam_name"
elif [[ "$bam_file" != *.bam ]]; then 
    echo "(read_sites.sh; Error) Input file is not a .bam file"
    exit 1
elif [[ ! -f "$out_file" ]]; then 
    # make path for dumping tsv files into
    mkdir -p $(dirname $out_file)
    # index bam file if not already indexed 
    bai_path="${bam_path}.bai"
    if [[ ! -f "$bai_path" ]]; then 
        echo "(Read Sites; Indexing bam file) $bam_name"
        eval "samtools index $bam_path"
    fi

    trunc_sites="${sites_file%.txt}_truc.txt"
    if [[ ! -f "$trunc_sites" ]]; then
        cut -f1-3 "$sites_file" > "$trunc_sites"
    fi

    echo "(Read Sites; Extracting Site Information) bam path: $bam_path "
    eval "bam-readcount -f $ref_gnm -l $trunc_sites $bam_path > $out_file"
    echo "(Read Sites; Extraction Complete) "
    
fi