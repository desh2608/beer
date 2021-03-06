#######################################################################
# Site specific configuration. Override these settings to run on
# your system.

hostname=$(hostname -f)
if [[ "$hostname" == *".fit.vutbr.cz" ]]; then
    timit=/mnt/matylda2/data/TIMIT/timit
    server=matylda5
    parallel_env=sge
    parallel_opts="-l mem_free=200M,ram_free=200M,$server=1"
    parallel_opts_gpu="-l gpu=1,mem_free=1G,ram_free=1G,hostname=*face*"
elif [[ "$hostname" = *"clsp.jhu.edu" ]]; then
    timit=/export/corpora5/LDC/LDC93S1/timit/TIMIT
    tr=/export/c03/draj/mozilla-common-voice/tr
    parallel_env=sge
    parallel_opts="-l mem_free=1G,ram_free=1G,hostname=b*|c*"
    parallel_opts_gpu="-l gpu=1,mem_free=1G,ram_free=1G,hostname=b1[123456789]*|c*"
else
    echo "Unkown location configuration. Please update the"
    echo "\"setup.sh\" file."
    exit 1
fi


#######################################################################
# Directory structure.

confdir=$(pwd)/conf
datadir=$(pwd)/data
langdir=$datadir/lang
expdir=$(pwd)/exp
corpus=tr

#######################################################################
# Features extraction.

fea_njobs=10
fea_parallel_opts="$parallel_opts"
fea_conf=$confdir/features.yml

#######################################################################
# AUD (HMM) model parameters.

aud_hmm_fea_type=mfcc
aud_hmm_model_name=aud_hmm
aud_hmm_conf=$confdir/${aud_hmm_model_name}/hmm-${corpus}.yml
aud_hmm_dir=$expdir/$aud_hmm_model_name
aud_hmm_n_units=100
aud_hmm_lm_concentration=1.
aud_hmm_align_njobs=10
aud_hmm_align_parallel_opts="$parallel_opts"
aud_hmm_align_iters="1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 23 26 29"
aud_hmm_train_iters=30
aud_hmm_train_lrate=5e-2
aud_hmm_train_batch_size=50
aud_hmm_train_epochs=10
aud_hmm_train_njobs=20
aud_hmm_train_opts="--fast-eval"
aud_hmm_train_parallel_opts="$parallel_opts"
aud_hmm_decode_njobs=10
aud_hmm_decode_parallel_opts="$parallel_opts"


#######################################################################
# AUD (VAE-HMM) model parameters.

aud_vae_hmm_fea_type=mfcc
aud_vae_hmm_model=aud_vae_hmm
aud_vae_hmm_confdir=$confdir/$aud_vae_hmm_model
aud_vae_hmm_dir=$expdir/$aud_vae_hmm_model
aud_vae_hmm_encoder_conf=$aud_vae_hmm_confdir/encoder.yml
aud_vae_hmm_decoder_conf=$aud_vae_hmm_confdir/decoder.yml
aud_vae_hmm_nflow_conf=$aud_vae_hmm_confdir/normalizing_flow.yml
aud_vae_hmm_hmm_conf=$aud_vae_hmm_confdir/hmm.yml
aud_vae_hmm_nnet_width=128
aud_vae_hmm_latent_dim=30
aud_vae_hmm_encoder_cov_type=isotropic
aud_vae_hmm_decoder_cov_type=diagonal
aud_vae_hmm_n_units=100
aud_vae_hmm_lm_concentration=1.
aud_vae_hmm_align_njobs=20
aud_vae_hmm_align_parallel_opts="$parallel_opts"
aud_vae_hmm_align_iters=$(seq 2 30)
aud_vae_hmm_train_warmup_iters=0
aud_vae_hmm_train_iters=30
aud_vae_hmm_train_epochs_per_iter=1
aud_vae_hmm_train_nnet_lrate=1e-3
aud_vae_hmm_train_lrate=1e-1
aud_vae_hmm_train_batch_size=50
aud_vae_hmm_train_epochs=30
aud_vae_hmm_train_opts="--fast-eval"
aud_vae_hmm_train_parallel_opts="$parallel_opts"
aud_vae_hmm_decode_njobs=2
aud_vae_hmm_decode_parallel_opts="$parallel_opts"


#######################################################################
# Score options.

remove_sym="" # Support multiple symbol, e.g. "sil spn nsn"
duplicate="no" # Do not allow adjacent duplicated phones. Only effective at scoring stage.

