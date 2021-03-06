#!/bin/bash

if [ $# -ne 4 ];then
    echo "$0 <setup.sh> <model-dir> <data-dir> <decode-dir> "
    exit 1
fi

setup=$1
mdldir=$2
data_dir=$3
fea_dir=$4
decode_dir=$5

. $setup
mdl=$mdldir/final.mdl
pdf_mapping=$mdldir/pdf_mapping.txt
fea=$data_train_dir/${aud_vae_hmm_fea_type}.npz

mkdir -p $decode_dir

# Load a numpy npz file and print its content as:
# utt1 pdf_id1 pdf_id2 ...
# utt2 pdf_id3 pdf_id4 ...
# ...
print_pdf_id="
import numpy as np
decoded = np.load('$decode_dir/best_paths.npz')
for utt in decoded.files:
    str_path = [str(pdf_id) for pdf_id in decoded[utt]]
    print(utt, ' '.join(str_path))
"

if [ ! -f $decode_dir/hyp ];then
    echo "Decoding..."

    tmpdir=$(mktemp -d $decode_dir/beer.XXXX);
    trap 'rm -rf "$tmpdir"' EXIT

    cmd="python utils/vae-hmm-decode-parallel.py $mdl \
        $fea $tmpdir"
    utils/parallel/submit_parallel.sh \
        "$parallel_env" \
        "vae-hmm-decode" \
        "$hmm_decode_parallel_opts" \
        $hmm_decode_njobs \
        $data_dir/uttids \
        "$cmd" \
        $decode_dir || exit 1
    find $tmpdir -name '*npy' | zip -j -@ $decode_dir/best_paths.npz \
        > /dev/null || exit 1


    python -c "$print_pdf_id" | \
        python utils/pdf2unit.py --phone-level $pdf_mapping  \
        > $decode_dir/hyp
    ln -s $data_dir/trans $decode_dir/trans
else
    echo "Decoding already done. Skipping."
fi

