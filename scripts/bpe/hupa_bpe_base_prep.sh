#!/bin/bash

numBpeMergeOps=$1
tier=$2
split=random
n=$3
lm_order=$4

#### Base ####

./subword-nmt/learn_bpe.py -s $numBpeMergeOps < data_lexicon/hupa/"$tier"_tier/"$split"/train$n/corpus > data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/corpus.$numBpeMergeOps.codeFile

# Let's create the train file
./subword-nmt/apply_bpe.py -c data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/corpus.$numBpeMergeOps.codeFile < data_lexicon/hupa/"$tier"_tier/"$split"/train$n/corpus > data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/corpus

python3 scripts/printFromTwoFiles.py -fileOne data_lexicon/hupa/"$tier"_tier/"$split"/train$n/text -fileTwo data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/corpus > data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/text

# Now let's create the test file
cut -d" " -f2- data_lexicon/hupa/"$tier"_tier/"$split"/dev$n/text > data_lexicon/hupa/"$tier"_tier/"$split"/dev$n/corpus

./subword-nmt/apply_bpe.py -c data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/corpus.$numBpeMergeOps.codeFile < data_lexicon/hupa/"$tier"_tier/"$split"/dev$n/corpus > data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/dev$n/corpus

python3 scripts/printFromTwoFiles.py -fileOne data_lexicon/hupa/"$tier"_tier/"$split"/dev$n/text -fileTwo data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/dev$n/corpus > data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/dev$n/text

#mkdir HupaLower_$numBpeMergeOps

#mv HupaLowerTrain$numBpeMergeOps.bpe HupaLowerTrainBpe.text$numBpeMergeOps HupaLowerTest$numBpeMergeOps.bpe HupaLowerTestBpe.text$numBpeMergeOps HupaLower_$numBpeMergeOps
 
#./subword-nmt/apply_bpe.py -c Hupa$numBpeMergeOps.codeFile < HupaFiles/fullHupaCorpus.txt > Corpus.bpe$numBpeMergeOps

#mv Hupa$numBpeMergeOps.codeFile HupaLower_$numBpeMergeOps

sed "s/ /\n/g" data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/corpus > data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/wordList
sort -u data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/wordList > data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/wordList.sorted

mkdir data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/local/
mkdir data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/local/dict/

echo "SIL" > data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/local/dict/optional_silence.txt

echo "SIL" > data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/local/dict/silence_phones.txt
echo "SPN" >> data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/local/dict/silence_phones.txt

python3 scripts/create_lexicon.py -wordList data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/wordList.sorted > data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/local/dict/lexicon.txt
python3 scripts/nonsilence_phones.py data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/local/dict/lexicon.txt > data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/local/dict/nonsilence_phones.txt

#mv Corpus.bpe$numBpeMergeOps HupaBpe$numBpeMergeOps.lexicon Bpe$numBpeMergeOps.wordList.sorted HupaLower_$numBpeMergeOps

#rm Bpe$numBpeMergeOps.wordList

### Making language models ###

bash path.sh

bash utils/prepare_lang.sh data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/local/dict "<UNK>" data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/local/lang data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/lang


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

local=data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/local

## Base language model, i.e., language model trained from only transcripts of training data
mkdir $local/tmp_base_"$lm_order"
ngram-count -order $lm_order -write-vocab $local/tmp_base_"$lm_order"/vocab-full.txt -wbdiscount -text data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/corpus -lm $local/tmp_base_"$lm_order"/lm.arpa
echo
echo "===== MAKING G.fst ====="
echo
original_lang=data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/lang

mkdir data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/lang_base_"$lm_order"
cp -R $original_lang/* data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/lang_base_"$lm_order"

lang=data_lexicon/hupa/"$tier"_tier/"$split"_base_bpe_$numBpeMergeOps/train$n/lang_base_"$lm_order"
src/lmbin/arpa2fst --disambig-symbol=#0 --read-symbol-table=$original_lang/words.txt $local/tmp_base_"$lm_order"/lm.arpa $lang/G.fst



