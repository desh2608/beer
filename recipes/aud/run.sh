#!/usr/bin/env bash

# Exit if one command fails.
set -e
stage=0
db=tr

. utils/parse_options.sh
#######################################################################
## SETUP

# Directory structure
datadir=data
feadir=features
expdir=exp

# Data
train_set=train

# Features
feaname=mfcc

# AUD
nunits=100      # maximum number of discovered units
epochs=10       # number of training epochs

#######################################################################

# Load the BEER anaconda environment.
. path.sh


# Create the directory structure.
mkdir -p $datadir $expdir $feadir

if [ $stage -le 0 ]; then
  echo "--> Preparing data for the $db database"
  local/$db/prepare_data.sh $datadir/$db
fi

if [ $stage -le 1 ]; then
  echo "--> Preparing pseudo-phones \"language\" information"
  mkdir -p data/$db/lang_aud
  # The option "non-speech-unit" will force the decoder to start and end
  # each utterance by a specific acoustic unit named "sil". The
  # unit can also be freely decoded within the utterance. If your data
  # is not well segmented and you are not sure that most of your data
  # start and end with non-speech sound you better remove this option.
  python utils/prepare_lang_aud.py \
      --non-speech-unit \
      $nunits > data/$db/lang_aud/units
fi

if [ $stage -le 2 ]; then
  for dataset in train dev test; do
    echo "--> Extracting features for the $db database"
    steps/extract_features.sh conf/${feaname}.yml $datadir/$db/$dataset \
           $feadir/$db/$dataset
  done
fi

if [ $stage -le 3 ]; then
  # Create a "dataset". This "dataset" is just an object
  # associating the features with their utterance id and some
  # other meta-data (e.g. global mean, variance, ...).
  for dataset in train dev test; do
    echo "--> Creating $dataset dataset(s) for $db database"
    steps/create_dataset.sh $datadir/$db/$dataset \
        $feadir/$db/$dataset/${feaname}.npz \
        $expdir/$db/datasets/$feaname/${dataset}.pkl
  done
fi

if [ $stage -le 4 ]; then
  # Train a GMM-HMM model for AUD
  steps/aud_hmm.sh \
      --parallel-opts "-l mem_free=1G,ram_free=1G" \
      --parallel-njobs 30 \
      conf/hmm-$db.yml \
      data/$db/lang_aud \
      data/$db/$train_set/uttids \
      $expdir/$db/datasets/$feaname/${train_set}.pkl \
      $epochs $expdir/$db/aud
fi

if [ $stage -le 5 ]; then
  # Decode and score the GMM-HMM model (score using boundary method)
	for dataset in dev test; do
    steps/decode.sh \
			--parallel-opts "-l mem_free=1G,ram_free=1G" \
			--parallel-njobs 4 \
			$expdir/$db/aud/final.mdl \
			data/$db/$dataset\
			$expdir/$db/datasets/$feaname/$dataset.pkl \
			$expdir/$db/aud/decode/$dataset 
  done

  for dataset in dev test; do
    echo "Scoring $dataset using boundary method"
    python utils/score_boundaries.py -d 0 data/$db/$dataset/trans \
      $expdir/$db/aud/decode/$dataset/trans
  done
fi

#if [ $stage -le 5 ]; then
  # Train an HMM-VAE model for AUD
#fi
