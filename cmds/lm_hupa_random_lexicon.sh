bash path.sh

tier=$1
n=$2

bash data_lexicon/hupa/"$tier"_tier/random/hupa_random_same_utt_spk$n.sh

utils/fix_data_dir.sh data_lexicon/hupa/"$tier"_tier/random/train$n
utils/fix_data_dir.sh data_lexicon/hupa/"$tier"_tier/random/dev$n

bash data_lexicon/hupa/"$tier"_tier/random/hupa_random_same_compute_mfcc$n.sh

#bash utils/prepare_lang.sh data_lexicon/hupa/local/dict "<UNK>" data_lexicon/hupa/local/lang data_lexicon/hupa/lang

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

sed "s/ /\n/g" data_lexicon/hupa/"$tier"_tier/random/train$n/corpus > data_lexicon/hupa/"$tier"_tier/random/train$n/wordList
sort -u data_lexicon/hupa/"$tier"_tier/random/train$n/wordList > data_lexicon/hupa/"$tier"_tier/random/train$n/wordList.sorted


mkdir data_lexicon/hupa/"$tier"_tier/random/train$n/local
mkdir data_lexicon/hupa/"$tier"_tier/random/train$n/local/dict_base/

local=data_lexicon/hupa/"$tier"_tier/random/train$n/local

echo "SIL" > $local/dict_base/optional_silence.txt

echo "SIL" > $local/dict_base/silence_phones.txt
echo "SPN" >> $local/dict_base/silence_phones.txt

python3 scripts/create_lexicon.py -wordList data_lexicon/hupa/"$tier"_tier/random/train$n/wordList.sorted > $local/dict_base/lexicon.txt
sort -u $local/dict_base/lexicon.txt > foo
mv foo $local/dict_base/lexicon.txt

python3 scripts/nonsilence_phones.py $local/dict_base/lexicon.txt > $local/dict_base/nonsilence_phones.txt

bash utils/prepare_lang.sh $local/dict_base/ "<UNK>" $local/lang_base data_lexicon/hupa/"$tier"_tier/random/train$n/lang_base


## Base language model, i.e., language model trained from only transcripts of training data
mkdir $local/tmp_base
ngram-count -order $lm_order -write-vocab $local/tmp_base/vocab-full.txt -wbdiscount -text data_lexicon/hupa/"$tier"_tier/random/train$n/corpus -lm $local/tmp_base/lm.arpa

#original_lang=data_lexicon/hupa/lang

#mkdir data_lexicon/hupa/"$tier"_tier/random/train$n/lang_base
#cp -R $original_lang/* data_lexicon/hupa/"$tier"_tier/random/train$n/lang_base/

lang_dir=data_lexicon/hupa/"$tier"_tier/random/train$n/lang_base
src/lmbin/arpa2fst --disambig-symbol=#0 --read-symbol-table=$lang_dir/words.txt $local/tmp_base/lm.arpa $lang_dir/G.fst


## Largest language model, i.e., language model trained from the concatenation of transcripts of training data and all the external text
cat data_lexicon/hupa/"$tier"_tier/random/train$n/corpus data_lexicon/hupa/local/hupa_texts.txt > data_lexicon/hupa/"$tier"_tier/random/train$n/local/corpus.txt

sed "s/ /\n/g" data_lexicon/hupa/"$tier"_tier/random/train$n/local/corpus.txt > data_lexicon/hupa/"$tier"_tier/random/train$n/wordList
sort -u data_lexicon/hupa/"$tier"_tier/random/train$n/wordList > data_lexicon/hupa/"$tier"_tier/random/train$n/wordList.sorted

mkdir data_lexicon/hupa/"$tier"_tier/random/train$n/local/dict_large/

local=data_lexicon/hupa/"$tier"_tier/random/train$n/local

echo "SIL" > $local/dict_large/optional_silence.txt

echo "SIL" > $local/dict_large/silence_phones.txt
echo "SPN" >> $local/dict_large/silence_phones.txt

python3 scripts/create_lexicon.py -wordList data_lexicon/hupa/"$tier"_tier/random/train$n/wordList.sorted > $local/dict_large/lexicon.txt
sort -u $local/dict_large/lexicon.txt > foo
mv foo $local/dict_large/lexicon.txt

python3 scripts/nonsilence_phones.py $local/dict_large/lexicon.txt > $local/dict_large/nonsilence_phones.txt

bash utils/prepare_lang.sh $local/dict_base/ "<UNK>" $local/lang_large data_lexicon/hupa/"$tier"_tier/random/train$n/lang_large

mkdir $local/tmp_large

ngram-count -order $lm_order -write-vocab $local/tmp_large/vocab-full.txt -wbdiscount -text $local/corpus.txt -lm $local/tmp_large/lm.arpa

#original_lang=data_lexicon/hupa/lang

#mkdir data_lexicon/hupa/"$tier"_tier/random/train$n/lang_large
#cp -R $original_lang/* data_lexicon/hupa/"$tier"_tier/random/train$n/lang_large/

lang_dir=data_lexicon/hupa/"$tier"_tier/random/train$n/lang_large
src/lmbin/arpa2fst --disambig-symbol=#0 --read-symbol-table=$lang_dir/words.txt $local/tmp_large/lm.arpa $lang_dir/G.fst
