bash path.sh

bash utils/prepare_lang.sh data/bemba/local/dict "<UNK>" data/bemba/local/lang data/bemba/lang

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

mkdir data/bemba/original/train1/local
local=data/bemba/original/train1/local

## Base language model, i.e., language model trained from only transcripts of training data
mkdir $local/tmp_base
ngram-count -order $lm_order -write-vocab $local/tmp_base/vocab-full.txt -wbdiscount -text data/bemba/local/lm1_corpus.txt -lm $local/tmp_base/lm.arpa
echo
echo "===== MAKING G.fst ====="
echo
original_lang=data/bemba/lang

mkdir data/bemba/original/train1/lang_base
cp -R $original_lang/* data/bemba/original/train1/lang_base/

lang=data/bemba/original/train1/lang_base
src/lmbin/arpa2fst --disambig-symbol=#0 --read-symbol-table=$original_lang/words.txt $local/tmp_base/lm.arpa $lang/G.fst


## Largest language model, i.e., language model trained from the concatenation of transcripts of training data and all the external text

mkdir $local/tmp_large

ngram-count -order $lm_order -write-vocab $local/tmp_large/vocab-full.txt -wbdiscount -text data/bemba/local/lm2_corpus.txt -lm $local/tmp_large/lm.arpa
echo
echo "===== MAKING G.fst ====="
echo
original_lang=data/bemba/lang

mkdir data/bemba/original/train1/lang_large
cp -R $original_lang/* data/bemba/original/train1/lang_large/

lang=data/bemba/original/train1/lang_large
src/lmbin/arpa2fst --disambig-symbol=#0 --read-symbol-table=$original_lang/words.txt $local/tmp_large/lm.arpa $lang/G.fst
