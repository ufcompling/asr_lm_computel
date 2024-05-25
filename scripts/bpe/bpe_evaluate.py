import io, os
from jiwer import wer

### ASR output
### exp_lexicon/fongbe/random_base_bpe_200/system1_base_5/dnn4b_pretrain-dbn_dnn_smbr/decode_dev_it6/log/decode.1.log

### Gold
### data_lexicon/fongbe/random_base_bpe_200/dev1/text

### Converg segmentation
### sed -r 's/(@@ )|(@@ ?$)//g' bpe_output > word_output

audios = []
gold_bpe = []
with open('data_lexicon/fongbe/random_base_bpe_200/dev1/text') as f:
	for line in f:
		toks = line.strip().split()
		audios.append(toks[0])
		gold_bpe.append(' '.join(w for w in toks[1 : ]))

with open('gold_bpe', 'w') as f:
	for utt in gold_bpe:
		f.write(utt + '\n')

os.system("sed -r 's/(@@ )|(@@ ?$)//g' gold_bpe > gold_word")

output = []
with open('exp_lexicon/fongbe/random_base_bpe_200/system1_base_5/dnn4b_pretrain-dbn_dnn_smbr/decode_dev_it6/log/decode.1.log') as f:
	for line in f:
		toks = line.strip().split()
		output.append(toks)

pred_bpe = []
for i in range(len(audios)):
	audio = audios[i]
	for toks in output:	
		if toks[0] == audio:
			print(toks)
			pred_bpe.append(' '.join(w for w in toks[1 : ]))

with open('pred_bpe', 'w') as f:
	for utt in pred_bpe:
		f.write(utt + '\n')

os.system("sed -r 's/(@@ )|(@@ ?$)//g' pred_bpe > pred_word")

gold_word = []
with open('gold_word') as f:
	for line in f:
		toks = line.strip()
		gold_word.append(toks)

pred_word = []
with open('pred_word') as f:
	for line in f:
		toks = line.strip()
		pred_word.append(toks)

wer = wer(gold_word, pred_word)
print(wer)
