import io, os, argparse, librosa
import pandas as pd

def get_text(file):
	texts = {}
	with open(file) as f:
		for line in f:
			toks = line.strip().split()
			utt = toks[0]
			text = ' '.join(w for w in toks[1 : ])
			texts[utt] = text
	return texts

def utt2spk(file):
	utt2spk_dict = {}
	with open(file) as f:
		for line in f:
			toks = line.strip().split()
			utt = toks[0]
			spk = toks[1]
			utt2spk_dict[utt] = spk
	return utt2spk_dict

train_data = []
test_data = []

train_texts = get_text('/data/liuaal/ALFFA_PUBLIC/ASR/AMHARIC/data/train/text')
train_utt2spk = utt2spk('/data/liuaal/ALFFA_PUBLIC/ASR/AMHARIC/data/train/utt2spk')
for utt, text in train_texts.items():
	path = '/data/liuaal/ALFFA_PUBLIC/ASR/AMHARIC/data/train/wav/' + utt + '.wav'
	duration = librosa.get_duration(filename = path)
	info = [utt, path, duration, text]
	train_data.append(info)

test_texts = get_text('/data/liuaal/ALFFA_PUBLIC/ASR/AMHARIC/data/test/text')
test_utt2spk = utt2spk('/data/liuaal/ALFFA_PUBLIC/ASR/AMHARIC/data/test/utt2spk')
for utt, text in test_texts.items():
	path = '/data/liuaal/ALFFA_PUBLIC/ASR/AMHARIC/data/test/wav/' + utt + '.wav'
	duration = librosa.get_duration(filename = path)
	info = [utt, path, duration, text]
	test_data.append(info)

header = ['File', 'Path', 'Duration', 'Transcript']
with open('/data/liuaal/asr_lm_size/data_lexicon/amharic/audio_info.txt', 'w') as f:
	f.write('\t'.join(w for w in header))
	for tok in train_data:
		f.write('\t'.join(str(w) for w in tok) + '\n')
	for tok in test_data:
		f.write('\t'.join(str(w) for w in tok) + '\n')




