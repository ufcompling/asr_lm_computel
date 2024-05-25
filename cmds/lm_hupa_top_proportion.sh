### For creating augmented setting (to match the LM proportion for second tier data)

bash path.sh

tier=top
n=$1
proportion=$2
proportion_n=$3

#bash data_lexicon/hupa/"$tier"_tier/random/hupa_random_same_utt_spk$n.sh

#utils/fix_data_dir.sh data_lexicon/hupa/"$tier"_tier/random/train$n
#utils/fix_data_dir.sh data_lexicon/hupa/"$tier"_tier/random/dev$n

#bash data_lexicon/hupa/"$tier"_tier/random/hupa_random_same_compute_mfcc$n.sh

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

local=data_lexicon/hupa/"$tier"_tier/random/train$n/local

## Language model by proportion 
cat data_lexicon/hupa/"$tier"_tier/random/train$n/corpus data_lexicon/hupa/"$tier"_tier/proportion_corpus."$proportion"."$proportion_n" > data_lexicon/hupa/"$tier"_tier/random/train$n/local/proportion_corpus."$proportion"."$proportion_n"

sed "s/ /\n/g" $local/proportion_corpus."$proportion"."$proportion_n" > data_lexicon/hupa/"$tier"_tier/random/train$n/wordList
sort -u data_lexicon/hupa/"$tier"_tier/random/train$n/wordList > data_lexicon/hupa/"$tier"_tier/random/train$n/wordList.sorted


mkdir data_lexicon/hupa/"$tier"_tier/random/train$n/local/dict_proportion"$proportion"."$proportion_n"/

echo "SIL" > $local/dict_proportion"$proportion"."$proportion_n"/optional_silence.txt

echo "SIL" > $local/dict_proportion"$proportion"."$proportion_n"/silence_phones.txt
echo "SPN" >> $local/dict_proportion"$proportion"."$proportion_n"/silence_phones.txt

python3 scripts/create_lexicon.py -wordList data_lexicon/hupa/"$tier"_tier/random/train$n/wordList.sorted > $local/dict_proportion"$proportion"."$proportion_n"/lexicon.txt
sort -u $local/dict_proportion"$proportion"."$proportion_n"/lexicon.txt > foo
mv foo $local/dict_proportion"$proportion"."$proportion_n"/lexicon.txt

python3 scripts/nonsilence_phones.py $local/dict_proportion"$proportion"."$proportion_n"/lexicon.txt > $local/dict_proportion"$proportion"."$proportion_n"/nonsilence_phones.txt

bash utils/prepare_lang.sh $local/dict_proportion"$proportion"."$proportion_n"/ "<UNK>" $local/lang_proportion"$proportion"."$proportion_n" data_lexicon/hupa/"$tier"_tier/random/train$n/lang_proportion"$proportion"."$proportion_n"


mkdir $local/tmp_proportion"$proportion"."$proportion_n"

ngram-count -order $lm_order -write-vocab $local/tmp_proportion"$proportion"."$proportion_n"/vocab-full.txt -wbdiscount -text $local/proportion_corpus."$proportion"."$proportion_n" -lm $local/tmp_proportion"$proportion"."$proportion_n"/lm.arpa

#mkdir data_lexicon/"$lang"/random/train$n/lang_proportion1/
#cp -R $original_lang/* data_lexicon/"$lang"/random/train$n/lang_proportion1/

lang_dir=data_lexicon/hupa/"$tier"_tier/random/train$n/lang_proportion"$proportion"."$proportion_n"
src/lmbin/arpa2fst --disambig-symbol=#0 --read-symbol-table=$lang_dir/words.txt $local/tmp_proportion"$proportion"."$proportion_n"/lm.arpa $lang_dir/G.fst


