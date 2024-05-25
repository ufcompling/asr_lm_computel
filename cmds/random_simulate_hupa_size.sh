#!/bin/bash

lang=$1
tier=$2
size=$3
n=$4

### Initializing paths ###

#rm -rf exp mfcc data/"$lang"/"$tier"_tier/"$n"/random/train1/spk2utt data/"$lang"/"$tier"_tier/"$n"/random/train1/cmvn.scp data/"$lang"/"$tier"_tier/"$n"/random/train1/feats.scp data/"$lang"/"$tier"_tier/"$n"/random/train1/split1 data/"$lang"/"$tier"_tier/"$n"/random/train1/split4 data/"$lang"/"$tier"_tier/"$n"/random/train1/split8 data/test/spk2utt data/test/cmvn.scp data/test/feats.scp data/test/split1 data/test/split4 data/test/split8 data/local/lang data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size" data/local/tmp data/local/dict/lexiconp.txt data/test_hires data/"$lang"/"$tier"_tier/"$n"/random/train1_sp data/"$lang"/"$tier"_tier/"$n"/random/train1_sp_hires data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size"_chain

#rm -rf mfcc data/"$lang"/"$tier"_tier/"$n"/random/train1/spk2utt data/"$lang"/"$tier"_tier/"$n"/random/train1/cmvn.scp data/"$lang"/"$tier"_tier/"$n"/random/train1/feats.scp data/"$lang"/"$tier"_tier/"$n"/random/train1/split1 data/"$lang"/"$tier"_tier/"$n"/random/train1/split4 data/"$lang"/"$tier"_tier/"$n"/random/train1/split8 data/test/spk2utt data/test/cmvn.scp data/test/feats.scp data/test/split1 data/test/split4 data/test/split8 data/local/lang data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size" data/local/tmp data/local/dict/lexiconp.txt data/test_hires data/"$lang"/"$tier"_tier/"$n"/random/train1_sp data/"$lang"/"$tier"_tier/"$n"/random/train1_sp_hires data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size"_chain

bash path.sh

bash kaldi_scripts/00_init_paths.sh

### Linking tools and scripts from public kaldi gpu folder on Sirius cluster ###

#bash kaldi-scripts/01_init_symlink.sh

### Checking lexicon ###

### Make sure to run lexicon_phones.py to generate nonsilence_phones.txt separately (and when adding data from second tier) ###

#bash kaldi-scripts/02_lexicon.sh

#utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size"

### Prepare data ###

#bash kaldi-scripts/04_data_prep.sh



train_cmd="run.pl --gpu 1"


# monophones
echo
echo "===== MONO TRAINING ====="
echo
# Training
steps/train_mono.sh --nj 1 --cmd "$train_cmd" data/"$lang"/"$tier"_tier/"$n"/random/train1 data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size" exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/mono
echo
echo "===== MONO DECODING ====="
echo
# Graph compilation
utils/mkgraph.sh --mono data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size" exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/mono exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/mono/graph
# Decoding
steps/decode.sh --nj 1 --cmd "$train_cmd" exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/mono/graph data/"$lang"/"$tier"_tier/"$n"/random/dev1 exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/mono/decode_dev
echo
echo "===== MONO ALIGNMENT ====="
echo
steps/align_si.sh --boost-silence 1.25 --nj 1 --cmd "$train_cmd" data/"$lang"/"$tier"_tier/"$n"/random/train1 data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size" exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/mono exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/mono_ali


## Triphone
echo
echo "===== TRI1 (first triphone pass) TRAINING ====="
echo
# Training
echo -e "triphones step \n"
steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 4200 40000 data/"$lang"/"$tier"_tier/"$n"/random/train1 data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size" exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/mono_ali exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri1
echo
echo "===== TRI1 (first triphone pass) DECODING ====="
echo
# Graph compilation
utils/mkgraph.sh data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size" exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri1 exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri1/graph
# Decoding
steps/decode.sh --nj 1 --cmd "$train_cmd" exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri1/graph data/"$lang"/"$tier"_tier/"$n"/random/dev1 exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri1/decode_dev
echo
echo "===== TRI1 (first triphone pass) ALIGNMENT ====="
echo
steps/align_si.sh --nj 1 --cmd "$train_cmd" data/"$lang"/"$tier"_tier/"$n"/random/train1 data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size" exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri1 exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri1_ali

## Triphone + Delta Delta
echo
echo "===== TRI2a (second triphone pass) TRAINING ====="
echo
# Training
steps/train_deltas.sh --cmd utils/run.pl 4200 40000  data/"$lang"/"$tier"_tier/"$n"/random/train1 data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size" exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri1_ali exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri2a
echo
echo "===== TRI2a (second triphone pass) DECODING ====="
echo
# Graph compilation
utils/mkgraph.sh data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size" exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri2a exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri2a/graph
# Decoding
steps/decode.sh --nj 1 --cmd utils/run.pl exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri2a/graph data/"$lang"/"$tier"_tier/"$n"/random/dev1 exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri2a/decode_dev
echo
echo "===== TRI2a (second triphone pass) ALIGNMENT ====="
echo
steps/align_si.sh --nj 1 --cmd utils/run.pl data/"$lang"/"$tier"_tier/"$n"/random/train1 data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size" exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri2a exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri2a_ali

## Triphone + Delta Delta + LDA and MLLT
echo
echo "===== TRI2b (third triphone pass) TRAINING ====="
echo
# Training
steps/train_lda_mllt.sh --cmd utils/run.pl --splice-opts "--left-context=3 --right-context=3"  4200 40000 data/"$lang"/"$tier"_tier/"$n"/random/train1 data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size" exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri2a_ali exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri2b
echo
echo "===== TRI2b (third triphone pass) DECODING ====="
echo
# Graph compilation
utils/mkgraph.sh data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size" exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri2b exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri2b/graph
# Decoding
steps/decode.sh --nj 1 --cmd utils/run.pl exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri2b/graph data/"$lang"/"$tier"_tier/"$n"/random/dev1 exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri2b/decode_dev
echo
echo "===== TRI2b (third triphone pass) ALIGNMENT ====="
echo
steps/align_si.sh --nj 1 --cmd utils/run.pl --use-graphs true data/"$lang"/"$tier"_tier/"$n"/random/train1 data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size" exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri2b exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri2b_ali

## Triphone + Delta Delta + LDA and MLLT + SAT and FMLLR
echo
echo "===== TRI3b (fourth triphone pass) TRAINING ====="
echo
# Training
steps/train_sat.sh --cmd utils/run.pl 4200 40000 data/"$lang"/"$tier"_tier/"$n"/random/train1 data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size" exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri2b_ali exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri3b
echo
echo "===== TRI3b (fourth triphone pass) DECODING ====="
echo
# Graph compilation
utils/mkgraph.sh data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size" exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri3b exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri3b/graph
# Decoding
steps/decode_fmllr.sh --nj 1 --cmd utils/run.pl exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri3b/graph data/"$lang"/"$tier"_tier/"$n"/random/dev1 exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri3b/decode_dev
echo
echo "===== TRI3b (fourth triphone pass) ALIGNMENT ====="
echo
# HMM/GMM aligments
steps/align_fmllr.sh --nj 1 --cmd utils/run.pl data/"$lang"/"$tier"_tier/"$n"/random/train1 data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size" exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri3b exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri3b_ali

echo
echo "===== CREATE DENOMINATOR LATTICES FOR MMI TRAINING ====="
echo
steps/make_denlats.sh --nj 1 --cmd utils/run.pl --sub-split 14 --transform-dir exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri3b_ali data/"$lang"/"$tier"_tier/"$n"/random/train1 data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size" exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri3b exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri3b_denlats || exit 1;

## Triphone + LDA and MLLT + SAT and FMLLR + fMMI and MMI
# Training
echo
echo "===== TRI3b_MMI (fifth triphone pass) TRAINING ====="
echo
steps/train_mmi.sh --cmd utils/run.pl --boost 0.1 data/"$lang"/"$tier"_tier/"$n"/random/train1 data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size" exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri3b_ali exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri3b_denlats exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri3b_mmi_b0.1  || exit 1;
# Decoding
echo
echo "===== TRI3b_MMI (fifth triphone pass) DECODING ====="
echo
steps/decode.sh --nj 1 --cmd utils/run.pl --transform-dir exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri3b/decode_dev exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri3b/graph data/"$lang"/"$tier"_tier/"$n"/random/dev1 exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri3b_mmi_b0.1/decode_dev

for x in exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/*/decode_*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done > exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/RESULTS


## DNN
echo
echo "===== DNN DATA PREPARATION ====="
echo
# Config:
gmmdir=/data/liuaal/asr_lm_size/exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri3b
data_fmllr=/data/liuaal/asr_lm_size/exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri3b
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
$dir data/"$lang"/"$tier"_tier/"$n"/random/dev1 $gmmdir $dir/log $dir/data || exit 1
# train
dir=$data_fmllr/train
bash /data/liuaal/asr_lm_size/steps/nnet/make_fmllr_feats.sh --nj 1 --cmd "$train_cmd" \
--transform-dir ${gmmdir}_ali \
$dir data/"$lang"/"$tier"_tier/"$n"/random/train1 $gmmdir $dir/log $dir/data || exit 1
# split the data : 90% train 10% cross-validation (held-out)
/data/liuaal/asr_lm_size/utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
fi
#  steps/nnet/make_fmllr_feats.sh --nj 1 --cmd "$train_cmd" \
#  steps/nnet/make_fmllr_feats.sh --nj 1 --cmd "$train_cmd" \

#cp pr exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri3b/dev/* exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri3b/train_cv10
#cp pr exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri3b/train/* exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/tri3b/train_tr10

echo
echo "===== DNN DATA TRAINING ====="
echo

# Training

if [ $stage -le 1 ]; then
# Pre-train DBN, i.e. a stack of RBMs (small database, smaller DNN)
dir=/data/liuaal/asr_lm_size/exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/dnn4b_pretrain-dbn
(tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
$cuda_cmd $dir/log/pretrain_dbn.log \
/data/liuaal/asr_lm_size/steps/nnet/pretrain_dbn.sh --hid-dim 1024 --rbm-iter 14 $data_fmllr/train $dir || exit 1;
fi

chmod -R 777 /data/liuaal/asr_lm_size/exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/dnn4b_pretrain-dbn_dnn/log/train_nnet.log

if [ $stage -le 2 ]; then
# Train the DNN optimizing per-frame cross-entropy.
dir=/data/liuaal/asr_lm_size/exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/dnn4b_pretrain-dbn_dnn
ali=${gmmdir}_ali
feature_transform=exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/dnn4b_pretrain-dbn/final.feature_transform
dbn=/data/liuaal/asr_lm_size/exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/dnn4b_pretrain-dbn/2.dbn
TRAIN_DIR=train
(tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
# Train
$cuda_cmd $dir/log/train_nnet.log \
/data/liuaal/asr_lm_size/steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
$data_fmllr/${TRAIN_DIR}_tr90 $data_fmllr/${TRAIN_DIR}_tr90 data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size" $ali $ali $dir || exit 1;
# Decode (reuse HCLG graph)
/data/liuaal/asr_lm_size/steps/nnet/decode.sh --nj 1 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
$gmmdir/graph $data_fmllr/dev $dir/decode_dev || exit 1;
fi

# Sequence training using sMBR criterion, we do Stochastic-GD
# with per-utterance updates. We use usually good acwt 0.1
dir=/data/liuaal/asr_lm_size/exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/dnn4b_pretrain-dbn_dnn_smbr
srcdir=/data/liuaal/asr_lm_size/exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/dnn4b_pretrain-dbn_dnn
acwt=0.1

if [ $stage -le 3 ]; then
# First we generate lattices and alignments:
/data/liuaal/asr_lm_size/steps/nnet/align.sh --nj 1 --cmd "$train_cmd" \
$data_fmllr/train data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size" $srcdir ${srcdir}_ali || exit 1;
/data/liuaal/asr_lm_size/steps/nnet/make_denlats.sh --nj 1 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
$data_fmllr/train data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size" $srcdir ${srcdir}_denlats || exit 1;
fi

if [ $stage -le 4 ]; then
# Re-train the DNN by 2 iterations of sMBR
/data/liuaal/asr_lm_size/steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 6 --acwt $acwt --do-smbr true \
$data_fmllr/train data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_"$size" $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
# Decode
for ITER in 1 2 3 4 5 6; do
/data/liuaal/asr_lm_size/steps/nnet/decode.sh --nj 1 --cmd "$decode_cmd" --config conf/decode_dnn.config \
--nnet $dir/${ITER}.nnet --acwt $acwt \
$gmmdir/graph $data_fmllr/dev $dir/decode_dev_it${ITER} || exit 1;
done
fi

echo
echo "===== See results in 'exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/RESULTS' ====="
echo

for x in exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/*/decode_*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done > exp/"$lang"/"$tier"_tier/"$n"/random/system1_"$size"/RESULTS


echo Success
exit 0

