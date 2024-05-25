#!/bin/bash

numBpeMergeOps=$1
lang=$2
split=random
n=$3
proportion=$4
proportion_n=$n
size=proportion"$proportion"."$proportion_n"

lm_order=$5

tier=$6

./subword-nmt/learn_bpe.py -s $numBpeMergeOps < data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"/train$n/local/proportion_corpus."$proportion"."$proportion_n" > data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/corpus.$numBpeMergeOps.codeFile

# Let's create the train file
./subword-nmt/apply_bpe.py -c data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/corpus.$numBpeMergeOps.codeFile < data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"/train$n/corpus > data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/corpus

python3 scripts/printFromTwoFiles.py -fileOne data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"/train$n/text -fileTwo data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/corpus > data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/text

# Now let's create the test file
cut -d" " -f2- data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"/dev$n/text > data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"/dev$n/corpus

./subword-nmt/apply_bpe.py -c data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/corpus.$numBpeMergeOps.codeFile < data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"/dev$n/corpus > data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/dev$n/corpus

python3 scripts/printFromTwoFiles.py -fileOne data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"/dev$n/text -fileTwo data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/dev$n/corpus > data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/dev$n/text


sed "s/ /\n/g" data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/corpus > data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/wordList
sort -u data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/wordList > data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/wordList.sorted

mkdir data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/local/
mkdir data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/local/dict/

echo "SIL" > data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/local/dict/optional_silence.txt

echo "SIL" > data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/local/dict/silence_phones.txt
echo "SPN" >> data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/local/dict/silence_phones.txt

python3 scripts/create_lexicon.py -wordList data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/wordList.sorted > data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/local/dict/lexicon.txt
python3 scripts/nonsilence_phones.py data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/local/dict/lexicon.txt > data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/local/dict/nonsilence_phones.txt


### Making language models ###

bash path.sh

bash utils/prepare_lang.sh data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/local/dict "<UNK>" data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/local/lang data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/lang



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

local=data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/local

## "$size" language model, i.e., language model trained from only transcripts of training data
mkdir $local/tmp_"$size"_"$lm_order"
ngram-count -order $lm_order -write-vocab $local/tmp_"$size"_"$lm_order"/vocab-full.txt -wbdiscount -text data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/corpus -lm $local/tmp_"$size"_"$lm_order"/lm.arpa
echo
echo "===== MAKING G.fst ====="
echo
original_lang=data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/lang

mkdir data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/lang_"$size"_"$lm_order"
cp -R $original_lang/* data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/lang_"$size"_"$lm_order"

lang=data_lexicon/"$lang"/"$tier"_tier/"$n"/"$split"_"$size"_bpe_$numBpeMergeOps/train$n/lang_"$size"_"$lm_order"
src/lmbin/arpa2fst --disambig-symbol=#0 --read-symbol-table=$original_lang/words.txt $local/tmp_"$size"_"$lm_order"/lm.arpa $lang/G.fst

