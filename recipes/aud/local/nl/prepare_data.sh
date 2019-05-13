#!/bin/bash

# Copyright 2019  Desh Raj 
# Apache 2.0.
#
# This is adapted from the TIMIT data preparation script.
#

set -e

# JHU
rootdir=/export/c03/draj/mozilla_common_voice

if [ $# -ne 1 ]; then
    echo "$0 <out-dir>"
    exit 1;
fi

if [ -d $rootdir/tr ]; then
  echo "$0: Corpus already extracted. Moving to data preparation."
else
  echo "$0: Extracting Turkish dataset.."
  tar -xzf $rootdir/nl.tar.gz 
fi

corpdir=$rootdir/nl/
conf=$(dirname $0)
outdir=$1
dir=$(pwd)/$outdir/local
mkdir -p $outdir/local/clips

[ -f $outdir/.done ] && echo "Data already prepared. Skipping." && exit 0;

for dataset in train dev test; do
  echo "Preparing $dataset data..."
  mkdir -p $outdir/$dataset
  total=$(< "${corpdir}/${dataset}.tsv" wc -l)
  counter=0
  sed 1d $corpdir/$dataset.tsv | while IFS=$'\t' read -r -a line
  do
    spkid="${line[0]}"
    uttid="${line[1]}"

    # Note that we need to save a copy in WAV format since sox cannot change header in a pipe
    ffmpeg -nostats -loglevel 0 -i $corpdir/clips/${uttid}.mp3 -f sox - | sox -p -t wav -r 16000 $outdir/local/clips/${uttid}.wav

    trans=$(echo "${line[2]}" | local/tr/get_ipa_phones.py)
    echo "$uttid" >> $outdir/$dataset/uttids
    echo "$uttid $outdir/local/clips/${uttid}.wav" >> $outdir/$dataset/wavs.scp
    echo "$uttid $trans" >> $outdir/$dataset/trans
    counter=$((counter+1))
    printf "\r%2f%s" "$(bc -l <<< "$counter*100/$total")" "% done."
  done
done

date > $outdir/.done
echo "Data preparation succeeded"

