bash path.sh

lang=$1
tier=$2
n=$3

bash data/"$lang"/"$tier"_tier/"$n"/random/"$lang"_random_same_utt_spk$n.sh

utils/fix_data_dir.sh data/"$lang"/"$tier"_tier/"$n"/random/train1
utils/fix_data_dir.sh data/"$lang"/"$tier"_tier/"$n"/random/dev1

bash data/"$lang"/"$tier"_tier/"$n"/random/"$lang"_random_same_compute_mfcc$n.sh

#bash utils/prepare_lang.sh data/"$lang"/local/dict "<UNK>" data/"$lang"/local/lang data/"$lang"/lang

lm_order=3

echo $lm_order

echo
echo "===== LANGUAGE MODEL CREATION ====="
echo "===== MAKING lm.arpa ====="
echo
loc=`which ngram-count`;
if [ -z $loc ]; then
        if uname -a | grep 64 >/dev/null; then
                sdir=$KALDI_ROOT/tools/srilm/bin/i686-m64
        else
                        sdir=$KALDI_ROOT/tools/srilm/bin/i686
        fi
        if [ -f $sdir/ngram-count ]; then
                        echo "Using SRILM language modelling tool from $sdir"
                        export PATH=$PATH:$sdir
        else
                        echo "SRILM toolkit is probably not installed.
                                Instructions: tools/install_srilm.sh"
                        exit 1
        fi
fi

mkdir data/"$lang"/"$tier"_tier/"$n"/random/train1/local
local=data/"$lang"/"$tier"_tier/"$n"/random/train1/local

## Base language model, i.e., language model trained from only transcripts of training data
mkdir $local/tmp_base
ngram-count -order $lm_order -write-vocab $local/tmp_base/vocab-full.txt -wbdiscount -text data/"$lang"/"$tier"_tier/"$n"/random/train1/corpus -lm $local/tmp_base/lm.arpa
echo
echo "===== MAKING G.fst ====="
echo
original_lang=data/"$lang"/lang

mkdir data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_base
cp -R $original_lang/* data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_base/

lang_dir=data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_base
src/lmbin/arpa2fst --disambig-symbol=#0 --read-symbol-table=$original_lang/words.txt $local/tmp_base/lm.arpa $lang_dir/G.fst


## Largest language model, i.e., language model trained from the concatenation of transcripts of training data and all the external text
cat data/"$lang"/"$tier"_tier/"$n"/random/train1/corpus data/"$lang"/local/corpus.txt > data/"$lang"/"$tier"_tier/"$n"/random/train1/local/corpus.txt

mkdir $local/tmp_large

ngram-count -order $lm_order -write-vocab $local/tmp_large/vocab-full.txt -wbdiscount -text $local/corpus.txt -lm $local/tmp_large/lm.arpa
echo
echo "===== MAKING G.fst ====="
echo
original_lang=data/"$lang"/lang

mkdir data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_large
cp -R $original_lang/* data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_large/

lang_dir=data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_large
src/lmbin/arpa2fst --disambig-symbol=#0 --read-symbol-table=$original_lang/words.txt $local/tmp_large/lm.arpa $lang_dir/G.fst

## Medium language model, i.e., language model trained on selected texts, 
## where the text size is the same as the additional texts for Hupa + transcripts of training audio data of the target language

mkdir $local/tmp_medium1

cat data/"$lang"/"$tier"_tier/"$n"/random/train1/corpus data/"$lang"/select_corpus.1 > data/"$lang"/"$tier"_tier/"$n"/random/train1/medium_corpus.1

ngram-count -order $lm_order -write-vocab $local/tmp_medium1/vocab-full.txt -wbdiscount -text data/"$lang"/"$tier"_tier/"$n"/random/train1/medium_corpus.1 -lm $local/tmp_medium1/lm.arpa
echo
echo "===== MAKING G.fst ====="
echo
original_lang=data/"$lang"/lang

mkdir data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_medium1
cp -R $original_lang/* data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_medium1/

lang_dir=data/"$lang"/"$tier"_tier/"$n"/random/train1/lang_medium1
src/lmbin/arpa2fst --disambig-symbol=#0 --read-symbol-table=$original_lang/words.txt $local/tmp_medium1/lm.arpa $lang_dir/G.fst


