#!/bin/bash

numBpeMergeOps=$1
lang=$2
n=$3
lm_order=$4
proportion=$5
proportion_n=$n
size=proportion"$proportion"."$proportion_n"
tier=$6

bash utt2spk_wav_simulate.sh $numBpeMergeOps $lang $n $size $tier

bash bpe_proportion_simulate_prep.sh $numBpeMergeOps $lang $n $proportion $lm_order $tier


bash path.sh

bash kaldi_scripts/00_init_paths.sh


train_cmd="run.pl --gpu 1"


# monophones
echo
echo "===== MONO TRAINING ====="
echo
# Training
steps/train_mono.sh --nj 1 --cmd "$train_cmd" data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_"$size"_"$lm_order" exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/mono
echo
echo "===== MONO DECODING ====="
echo
# Graph compilation
utils/mkgraph.sh --mono data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_"$size"_"$lm_order" exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/mono exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/mono/graph
# Decoding
steps/decode.sh --nj 1 --cmd "$train_cmd" exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/mono/graph data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/dev$n exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/mono/decode_dev
echo
echo "===== MONO ALIGNMENT ====="
echo
steps/align_si.sh --boost-silence 1.25 --nj 1 --cmd "$train_cmd" data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_"$size"_"$lm_order" exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/mono exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/mono_ali


## Triphone
echo
echo "===== TRI1 (first triphone pass) TRAINING ====="
echo
# Training
echo -e "triphones step \n"
steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 4200 40000 data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_"$size"_"$lm_order" exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/mono_ali exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri1
echo
echo "===== TRI1 (first triphone pass) DECODING ====="
echo
# Graph compilation
utils/mkgraph.sh data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_"$size"_"$lm_order" exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri1 exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri1/graph
# Decoding
steps/decode.sh --nj 1 --cmd "$train_cmd" exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri1/graph data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/dev$n exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri1/decode_dev
echo
echo "===== TRI1 (first triphone pass) ALIGNMENT ====="
echo
steps/align_si.sh --nj 1 --cmd "$train_cmd" data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_"$size"_"$lm_order" exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri1 exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri1_ali

## Triphone + Delta Delta
echo
echo "===== TRI2a (second triphone pass) TRAINING ====="
echo
# Training
steps/train_deltas.sh --cmd utils/run.pl 4200 40000  data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_"$size"_"$lm_order" exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri1_ali exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri2a
echo
echo "===== TRI2a (second triphone pass) DECODING ====="
echo
# Graph compilation
utils/mkgraph.sh data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_"$size"_"$lm_order" exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri2a exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri2a/graph
# Decoding
steps/decode.sh --nj 1 --cmd utils/run.pl exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri2a/graph data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/dev$n exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri2a/decode_dev
echo
echo "===== TRI2a (second triphone pass) ALIGNMENT ====="
echo
steps/align_si.sh --nj 1 --cmd utils/run.pl data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_"$size"_"$lm_order" exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri2a exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri2a_ali

## Triphone + Delta Delta + LDA and MLLT
echo
echo "===== TRI2b (third triphone pass) TRAINING ====="
echo
# Training
steps/train_lda_mllt.sh --cmd utils/run.pl --splice-opts "--left-context=3 --right-context=3"  4200 40000 data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_"$size"_"$lm_order" exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri2a_ali exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri2b
echo
echo "===== TRI2b (third triphone pass) DECODING ====="
echo
# Graph compilation
utils/mkgraph.sh data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_"$size"_"$lm_order" exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri2b exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri2b/graph
# Decoding
steps/decode.sh --nj 1 --cmd utils/run.pl exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri2b/graph data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/dev$n exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri2b/decode_dev
echo
echo "===== TRI2b (third triphone pass) ALIGNMENT ====="
echo
steps/align_si.sh --nj 1 --cmd utils/run.pl --use-graphs true data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_"$size"_"$lm_order" exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri2b exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri2b_ali

## Triphone + Delta Delta + LDA and MLLT + SAT and FMLLR
echo
echo "===== TRI3b (fourth triphone pass) TRAINING ====="
echo
# Training
steps/train_sat.sh --cmd utils/run.pl 4200 40000 data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_"$size"_"$lm_order" exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri2b_ali exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri3b
echo
echo "===== TRI3b (fourth triphone pass) DECODING ====="
echo
# Graph compilation
utils/mkgraph.sh data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_"$size"_"$lm_order" exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri3b exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri3b/graph
# Decoding
steps/decode_fmllr.sh --nj 1 --cmd utils/run.pl exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri3b/graph data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/dev$n exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri3b/decode_dev
echo
echo "===== TRI3b (fourth triphone pass) ALIGNMENT ====="
echo
# HMM/GMM aligments
steps/align_fmllr.sh --nj 1 --cmd utils/run.pl data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_"$size"_"$lm_order" exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri3b exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri3b_ali

echo
echo "===== CREATE DENOMINATOR LATTICES FOR MMI TRAINING ====="
echo
steps/make_denlats.sh --nj 1 --cmd utils/run.pl --sub-split 14 --transform-dir exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri3b_ali data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_"$size"_"$lm_order" exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri3b exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri3b_denlats || exit 1;

## Triphone + LDA and MLLT + SAT and FMLLR + fMMI and MMI
# Training
echo
echo "===== TRI3b_MMI (fifth triphone pass) TRAINING ====="
echo
steps/train_mmi.sh --cmd utils/run.pl --boost 0.1 data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_"$size"_"$lm_order" exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri3b_ali exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri3b_denlats exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri3b_mmi_b0.1  || exit 1;
# Decoding
echo
echo "===== TRI3b_MMI (fifth triphone pass) DECODING ====="
echo
steps/decode.sh --nj 1 --cmd utils/run.pl --transform-dir exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri3b/decode_dev exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri3b/graph data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/dev$n exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri3b_mmi_b0.1/decode_dev

for x in exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/*/decode_*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done > exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/RESULTS


## DNN
echo
echo "===== DNN DATA PREPARATION ====="
echo
# Config:
gmmdir=/data/liuaal/asr_lm_size/exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri3b
data_fmllr=/data/liuaal/asr_lm_size/exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri3b
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
$dir data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/dev$n $gmmdir $dir/log $dir/data || exit 1
# train
dir=$data_fmllr/train
bash /data/liuaal/asr_lm_size/steps/nnet/make_fmllr_feats.sh --nj 1 --cmd "$train_cmd" \
--transform-dir ${gmmdir}_ali \
$dir data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n $gmmdir $dir/log $dir/data || exit 1
# split the data : 90% train 10% cross-validation (held-out)
/data/liuaal/asr_lm_size/utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
fi
#  steps/nnet/make_fmllr_feats.sh --nj 1 --cmd "$train_cmd" \
#  steps/nnet/make_fmllr_feats.sh --nj 1 --cmd "$train_cmd" \

#cp pr exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri3b/dev/* exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri3b/train_cv10
#cp pr exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri3b/train/* exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/tri3b/train_tr10

echo
echo "===== DNN DATA TRAINING ====="
echo

# Training

if [ $stage -le 1 ]; then
# Pre-train DBN, i.e. a stack of RBMs (small database, smaller DNN)
dir=/data/liuaal/asr_lm_size/exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/dnn4b_pretrain-dbn
(tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
$cuda_cmd $dir/log/pretrain_dbn.log \
/data/liuaal/asr_lm_size/steps/nnet/pretrain_dbn.sh --hid-dim 1024 --rbm-iter 14 $data_fmllr/train $dir || exit 1;
fi

chmod -R 777 /data/liuaal/asr_lm_size/exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/dnn4b_pretrain-dbn_dnn/log/train_nnet.log

if [ $stage -le 2 ]; then
# Train the DNN optimizing per-frame cross-entropy.
dir=/data/liuaal/asr_lm_size/exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/dnn4b_pretrain-dbn_dnn
ali=${gmmdir}_ali
feature_transform=exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/dnn4b_pretrain-dbn/final.feature_transform
dbn=/data/liuaal/asr_lm_size/exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/dnn4b_pretrain-dbn/2.dbn
TRAIN_DIR=train
(tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
# Train
$cuda_cmd $dir/log/train_nnet.log \
/data/liuaal/asr_lm_size/steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
$data_fmllr/${TRAIN_DIR}_tr90 $data_fmllr/${TRAIN_DIR}_tr90 data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_"$size"_"$lm_order" $ali $ali $dir || exit 1;
# Decode (reuse HCLG graph)
/data/liuaal/asr_lm_size/steps/nnet/decode.sh --nj 1 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
$gmmdir/graph $data_fmllr/dev $dir/decode_dev || exit 1;
fi

# Sequence training using sMBR criterion, we do Stochastic-GD
# with per-utterance updates. We use usually good acwt 0.1
dir=/data/liuaal/asr_lm_size/exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/dnn4b_pretrain-dbn_dnn_smbr
srcdir=/data/liuaal/asr_lm_size/exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/dnn4b_pretrain-dbn_dnn
acwt=0.1

if [ $stage -le 3 ]; then
# First we generate lattices and alignments:
/data/liuaal/asr_lm_size/steps/nnet/align.sh --nj 1 --cmd "$train_cmd" \
$data_fmllr/train data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_"$size"_"$lm_order" $srcdir ${srcdir}_ali || exit 1;
/data/liuaal/asr_lm_size/steps/nnet/make_denlats.sh --nj 1 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
$data_fmllr/train data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_"$size"_"$lm_order" $srcdir ${srcdir}_denlats || exit 1;
fi

if [ $stage -le 4 ]; then
# Re-train the DNN by 2 iterations of sMBR
/data/liuaal/asr_lm_size/steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 6 --acwt $acwt --do-smbr true \
$data_fmllr/train data_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/train$n/lang_"$size"_"$lm_order" $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
# Decode
for ITER in 1 2 3 4 5 6; do
/data/liuaal/asr_lm_size/steps/nnet/decode.sh --nj 1 --cmd "$decode_cmd" --config conf/decode_dnn.config \
--nnet $dir/${ITER}.nnet --acwt $acwt \
$gmmdir/graph $data_fmllr/dev $dir/decode_dev_it${ITER} || exit 1;
done
fi

echo
echo "===== See results in 'exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/RESULTS' ====="
echo

for x in exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/*/decode_*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done > exp_lexicon/"$lang"/"$tier"_tier/"$n"/random_"$size"_bpe_$numBpeMergeOps/system"$n"_"$size"_"$lm_order"/RESULTS


echo Success
exit 0

