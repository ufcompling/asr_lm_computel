#!/bin/bash

numBpeMergeOps=$1
tier=$2
size=$3
n=$4

### Initializing paths ###

#rm -rf exp mfcc data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n/spk2utt data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n/cmvn.scp data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n/feats.scp data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n/split1 data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n/split4 data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n/split8 data/test/spk2utt data/test/cmvn.scp data/test/feats.scp data/test/split1 data/test/split4 data/test/split8 data/local/lang data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_""$size"" data/local/tmp data/local/dict/lexiconp.txt data/test_hires data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n_sp data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n_sp_hires data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_""$size""_chain

#rm -rf mfcc data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n/spk2utt data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n/cmvn.scp data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n/feats.scp data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n/split1 data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n/split4 data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n/split8 data/test/spk2utt data/test/cmvn.scp data/test/feats.scp data/test/split1 data/test/split4 data/test/split8 data/local/lang data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_""$size"" data/local/tmp data/local/dict/lexiconp.txt data/test_hires data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n_sp data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n_sp_hires data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_""$size""_chain

bash path.sh

bash kaldi_scripts/00_init_paths.sh

### Linking tools and scripts from public kaldi gpu folder on Sirius cluster ###

#bash kaldi-scripts/01_init_symlink.sh

### Checking lexicon ###

### Make sure to run lexicon_phones.py to generate nonsilence_phones.txt separately (and when adding data from second tier) ###

#bash kaldi-scripts/02_lexicon.sh

#utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_""$size""

### Prepare data ###

#bash kaldi-scripts/04_data_prep.sh

#bash kaldi_scripts/"$tier"_tier/hupa_random_same_utt_spk1.sh
#bash kaldi_scripts/"$tier"_tier/hupa_random_same_compute_mfcc1.sh


train_cmd="run.pl --gpu 1"


## DNN
echo
echo "===== DNN DATA PREPARATION ====="
echo
# Config:
gmmdir=/data/liuaal/asr_lm_size/exp/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/system$n_""$size""/tri3b
data_fmllr=/data/liuaal/asr_lm_size/exp/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/system$n_""$size""/tri3b
stage=0 # resume training with --stage=N
train_cmd="/data/liuaal/asr_lm_size/utils/run.pl --gpu 1"
cuda_cmd="/data/liuaal/asr_lm_size/utils/run.pl --gpu 1"
decode_cmd="/data/liuaal/asr_lm_size/utils/run.pl --gpu 1"
# End of config.
bash /data/liuaal/asr_lm_size/utils/parse_options.sh || exit 1;
#

if [ $stage -le 0 ]; then
# Store fMLLR features, so we can train on them easily,
# dev
dir=$data_fmllr/dev
bash /data/liuaal/asr_lm_size/steps/nnet/make_fmllr_feats.sh --nj 1 --cmd "$train_cmd" \
--transform-dir $gmmdir/decode_dev \
$dir data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/dev$n $gmmdir $dir/log $dir/data || exit 1
# train
dir=$data_fmllr/train
bash /data/liuaal/asr_lm_size/steps/nnet/make_fmllr_feats.sh --nj 1 --cmd "$train_cmd" \
--transform-dir ${gmmdir}_ali \
$dir data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n $gmmdir $dir/log $dir/data || exit 1
# split the data : 90% train 10% cross-validation (held-out)
/data/liuaal/asr_lm_size/utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
fi
#  steps/nnet/make_fmllr_feats.sh --nj 1 --cmd "$train_cmd" \
#  steps/nnet/make_fmllr_feats.sh --nj 1 --cmd "$train_cmd" \

#cp pr exp/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/system$n_""$size""/tri3b/dev/* exp/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/system$n_""$size""/tri3b/train_cv10
#cp pr exp/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/system$n_""$size""/tri3b/train/* exp/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/system$n_""$size""/tri3b/train_tr10

echo
echo "===== DNN DATA TRAINING ====="
echo

# Training

if [ $stage -le 1 ]; then
# Pre-train DBN, i.e. a stack of RBMs (small database, smaller DNN)
dir=/data/liuaal/asr_lm_size/exp/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/system$n_""$size""/dnn4b_pretrain-dbn
(tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
$cuda_cmd $dir/log/pretrain_dbn.log \
/data/liuaal/asr_lm_size/steps/nnet/pretrain_dbn.sh --hid-dim 1024 --rbm-iter 14 $data_fmllr/train $dir || exit 1;
fi

chmod -R 777 /data/liuaal/asr_lm_size/exp/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/system$n_""$size""/dnn4b_pretrain-dbn_dnn/log/train_nnet.log

if [ $stage -le 2 ]; then
# Train the DNN optimizing per-frame cross-entropy.
dir=/data/liuaal/asr_lm_size/exp/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/system$n_""$size""/dnn4b_pretrain-dbn_dnn
ali=${gmmdir}_ali
feature_transform=exp/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/system$n_""$size""/dnn4b_pretrain-dbn/final.feature_transform
dbn=/data/liuaal/asr_lm_size/exp/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/system$n_""$size""/dnn4b_pretrain-dbn/2.dbn
TRAIN_DIR=train
(tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
# Train
$cuda_cmd $dir/log/train_nnet.log \
/data/liuaal/asr_lm_size/steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
$data_fmllr/${TRAIN_DIR}_tr90 $data_fmllr/${TRAIN_DIR}_tr90 data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_""$size"" $ali $ali $dir || exit 1;
# Decode (reuse HCLG graph)
/data/liuaal/asr_lm_size/steps/nnet/decode.sh --nj 1 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
$gmmdir/graph $data_fmllr/dev $dir/decode_dev || exit 1;
fi

# Sequence training using sMBR criterion, we do Stochastic-GD
# with per-utterance updates. We use usually good acwt 0.1
dir=/data/liuaal/asr_lm_size/exp/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/system$n_""$size""/dnn4b_pretrain-dbn_dnn_smbr
srcdir=/data/liuaal/asr_lm_size/exp/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/system$n_""$size""/dnn4b_pretrain-dbn_dnn
acwt=0.1

if [ $stage -le 3 ]; then
# First we generate lattices and alignments:
/data/liuaal/asr_lm_size/steps/nnet/align.sh --nj 1 --cmd "$train_cmd" \
$data_fmllr/train data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_""$size"" $srcdir ${srcdir}_ali || exit 1;
/data/liuaal/asr_lm_size/steps/nnet/make_denlats.sh --nj 1 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
$data_fmllr/train data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_""$size"" $srcdir ${srcdir}_denlats || exit 1;
fi

if [ $stage -le 4 ]; then
# Re-train the DNN by 2 iterations of sMBR
/data/liuaal/asr_lm_size/steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 6 --acwt $acwt --do-smbr true \
$data_fmllr/train data/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_""$size"" $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
# Decode
for ITER in 1 2 3 4 5 6; do
/data/liuaal/asr_lm_size/steps/nnet/decode.sh --nj 1 --cmd "$decode_cmd" --config conf/decode_dnn.config \
--nnet $dir/${ITER}.nnet --acwt $acwt \
$gmmdir/graph $data_fmllr/dev $dir/decode_dev_it${ITER} || exit 1;
done
fi

echo
echo "===== See results in 'exp/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/system$n_""$size""/RESULTS' ====="
echo

for x in exp/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/system$n_""$size""/*/decode_*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done > exp/hupa/"$tier"_tier/random_"$size"_bpe_$numBpeMergeOps/system$n_""$size""/RESULTS


echo Success
exit 0

