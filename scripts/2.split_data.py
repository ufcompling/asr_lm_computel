import io, os, argparse, random, librosa, statistics
from pydub import AudioSegment
import parselmouth
import numpy as np
import seaborn as sns

from collections import Counter
from scipy.stats import wasserstein_distance

import collections
import random
from typing import Dict, Generator, Iterator, List, Set, Text, Tuple

import pandas as pd
from scipy import stats
from sklearn import feature_extraction
from sklearn import neighbors

HELDOUT_RATE = 0.2

### Read audio information file ###

def gather_audio_info(file):

	data = []

	with io.open(file, encoding = 'utf-8') as f:
		for line in f:
			toks = line.strip().split('\t')
			if toks[0].startswith('File') is False or toks[0] != 'File':
				data.append(toks)

	return data

### Get corpus from training data ###

def text_to_corpus(output, split_id):

	train_corpus = io.open(output + 'train' + split_id + '/corpus', 'w', encoding = 'utf-8')

	with io.open(output + 'train' + split_id + '/text', encoding = 'utf-8') as f:
		for line in f:
			toks = line.strip().split()
			transcripts = toks[1 : ]
			transcripts = ' '.join(w for w in transcripts)
			train_corpus.write(transcripts + '\n')

	dev_corpus = io.open(output + 'dev' + split_id + '/corpus', 'w', encoding = 'utf-8')

	with io.open(output + 'dev' + split_id + '/text', encoding = 'utf-8') as f:
		for line in f:
			toks = line.strip().split()
			transcripts = toks[1 : ]
			transcripts = ' '.join(w for w in transcripts)
			dev_corpus.write(transcripts + '\n')

def write_text_wav(train_texts, train_wav, dev_texts, dev_wav, split_id, output):

	with io.open(output + 'train' + split_id + '/text', 'w', encoding = 'utf-8') as f:
		for tok in train_texts:
			f.write(tok + '\n')

	with io.open(output + 'train' + split_id + '/wav.scp', 'w', encoding = 'utf-8') as f:
		for tok in train_wav:
			f.write(tok + '\n')

	with io.open(output + 'dev' + split_id + '/text', 'w', encoding = 'utf-8') as f:
		for tok in dev_texts:
			f.write(tok + '\n')

	with io.open(output + 'dev' + split_id + '/wav.scp', 'w', encoding = 'utf-8') as f:
		for tok in dev_wav:
			f.write(tok + '\n')

### Sorting order of audios for utt2spk and spk2utt ###

def sort_audio_id(audio_list):

	new_audio_list = []
	data = {}
	new_idx = []
	all_speakers = []

	tok = audio_list[0].split('_')
	lang = tok[0]
	misc = tok[2]

	for tok in audio_list:
		audio = tok.split('_')
		speaker = audio[1]

		if speaker.startswith('0'):
			speaker = int(speaker[-1])

		all_speakers.append(int(speaker))

	all_speakers = list(set(all_speakers))
	all_speakers.sort()

	for i in range(len(all_speakers)):
		speaker = all_speakers[i]

		if len(str(speaker)) == 1:
			speaker = '0' + str(speaker)
		else:
			speaker = str(speaker)

		audio_idx_list = []

		for tok in audio_list:
			audio = tok.split('_')
			audio_speaker = audio[1]

			if audio_speaker == speaker:
				idx = audio[-1]
				
				if idx.startswith('000'):
					idx = int(idx[-1])
				elif idx.startswith('00'):
					idx = int(idx[-2 : ])
				elif idx.startswith('0'):
					idx = int(idx[-3 : ])
				else:
					idx = int(idx)

				audio_idx_list.append(idx)

		audio_idx_list.sort()

		new_audio_idx_list = []

		for z in range(len(audio_idx_list)):
			name = audio
			temp_idx = audio_idx_list[z]
			if len(str(temp_idx)) == 1:
				new_idx = '000' + str(temp_idx)
			if len(str(temp_idx)) == 2:
				new_idx = '00' + str(temp_idx)
			if len(str(temp_idx)) == 3:
				new_idx = '0' + str(temp_idx)
			if len(str(temp_idx)) == 4:
				new_idx = str(temp_idx)

			new_name = lang + '_' + speaker + '_' + misc + '_' + new_idx

			new_audio_list.append(new_name)

	return new_audio_list

def iban_sort_audio_id(audio_list):

	new_audio_list = []
	data = {}
	new_idx = []
	all_speakers = []
	all_speaker_ids = []

	tok = audio_list[0].split('_')

	for tok in audio_list:
		audio = tok.split('_')
		speaker = audio[0] + '_' + audio[1]
		speaker_id = audio[1]

		if speaker_id.startswith('00'):
			speaker_id = int(speaker[-1])
		elif speaker_id.startswith('0'):
			speaker_id = int(speaker[-2 : ])

		all_speakers.append(speaker)
		all_speaker_ids.append(int(speaker_id))

	all_speakers = list(set(all_speakers))
	all_speaker_ids = list(set(all_speaker_ids))
	all_speaker_ids.sort()

	for i in range(len(all_speaker_ids)):
		speaker_id = all_speaker_ids[i]

		if len(str(speaker_id)) == 1:
			speaker_id = '00' + str(speaker_id)
		else:
			speaker_id = '0' + str(speaker_id)

		audio_idx_list = []

		for tok in audio_list:
			if tok.startswith('ibf'):
				audio = tok.split('_')
				audio_speaker = audio[1]

				if audio_speaker == speaker_id:
					idx = audio[-1]
				
					if idx.startswith('00'):
						idx = int(idx[-1])
					elif idx.startswith('0'):
						idx = int(idx[-2 : ])
					else:
						idx = int(idx)

					audio_idx_list.append(idx)

		audio_idx_list = list(set(audio_idx_list))
		audio_idx_list.sort()

		new_audio_idx_list = []

		for z in range(len(audio_idx_list)):
			temp_idx = audio_idx_list[z]
			if len(str(temp_idx)) == 1:
				new_idx = '00' + str(temp_idx)
			if len(str(temp_idx)) == 2:
				new_idx = '0' + str(temp_idx)
			if len(str(temp_idx)) == 3:
				new_idx = str(temp_idx)

			new_name = 'ibf_' + speaker_id + '_' + new_idx

			if new_name in audio_list and new_name not in new_audio_list:
				new_audio_list.append(new_name)

	for i in range(len(all_speaker_ids)):
		speaker_id = all_speaker_ids[i]

		if len(str(speaker_id)) == 1:
			speaker_id = '00' + str(speaker_id)
		else:
			speaker_id = '0' + str(speaker_id)

		audio_idx_list = []

		for tok in audio_list:
			if tok.startswith('ibm'):
				audio = tok.split('_')
				audio_speaker = audio[1]

				if audio_speaker == speaker_id:
					idx = audio[-1]
				
					if idx.startswith('00'):
						idx = int(idx[-1])
					elif idx.startswith('0'):
						idx = int(idx[-2 : ])
					else:
						idx = int(idx)

					audio_idx_list.append(idx)

		audio_idx_list = list(set(audio_idx_list))
		audio_idx_list.sort()

		new_audio_idx_list = []

		for z in range(len(audio_idx_list)):
			temp_idx = audio_idx_list[z]
			if len(str(temp_idx)) == 1:
				new_idx = '00' + str(temp_idx)
			if len(str(temp_idx)) == 2:
				new_idx = '0' + str(temp_idx)
			if len(str(temp_idx)) == 3:
				new_idx = str(temp_idx)

			new_name = 'ibm_' + speaker_id + '_' + new_idx

			if new_name in audio_list and new_name not in new_audio_list:
				new_audio_list.append(new_name)

	return new_audio_list

def fongbe_sort_audio_id(audio_list):

	new_audio_list = []
	data = {}
	new_idx = []
	all_speakers = []

	tok = audio_list[0].split('_')

	for tok in audio_list:
		audio = tok.split('_')
		speaker = audio[0]

		all_speakers.append(speaker)

	all_speakers = list(set(all_speakers))

	for speaker in sorted(all_speakers):
		corpora_list = []

		for tok in audio_list:
			audio = tok.split('_')
			corpus_id = audio[2][4 : ]

			assert len(corpus_id) == 3

			if corpus_id.startswith('00'):
				corpus_id = int(corpus_id[-1])
			elif corpus_id.startswith('0'):
				corpus_id = int(corpus_id[-2 : ])

			corpora_list.append(corpus_id)

		corpora_list = list(set(corpora_list))
		corpora_list.sort()

		for corpus in corpora_list:
			if len(str(corpus)) == 1:
				corpus = '00' + str(corpus)
			if len(str(corpus)) == 2:
				corpus = '0' + str(corpus)

			audio_idx_list = []

			for tok in audio_list:
				if tok.startswith(speaker + '_fongbe_corp' + str(corpus)):
					audio = tok.split('_')
					idx = audio[-1]
					if idx.startswith('00'):
						idx = int(idx[-1])
					elif idx.startswith('0'):
						idx = int(idx[-2 : ])
					else:
						idx = int(idx)

					audio_idx_list.append(idx)

			audio_idx_list = list(set(audio_idx_list))
			audio_idx_list.sort()

			new_audio_idx_list = []

			for z in range(len(audio_idx_list)):
				temp_idx = audio_idx_list[z]
				if len(str(temp_idx)) == 1:
					new_idx = '00' + str(temp_idx)
				if len(str(temp_idx)) == 2:
					new_idx = '0' + str(temp_idx)
				if len(str(temp_idx)) == 3:
					new_idx = str(temp_idx)

				new_name = speaker + '_fongbe_corp' + str(corpus) + '_' + new_idx

				if new_name in audio_list and new_name not in new_audio_list:
					new_audio_list.append(new_name)

	return new_audio_list

def swahili_sort_audio_id(audio_list, utt2spk_file):

	new_audio_list = []

	utt2spk_info = []
	with io.open(utt2spk_file, encoding = 'utf-8') as f:
		for line in f:
			toks = line.strip().split()
			utt2spk_info.append(toks)

	for i in range(len(utt2spk_info)):
		utt2spk = utt2spk_info[i]
		if utt2spk[0] in audio_list:
			new_audio_list.append(utt2spk[0])

	return new_audio_list


### Heldout one speaker ###

def heldout_speaker(audio_info_data, output, lang, utt2spk_file = None):

	time = 0
	all_audios = []
	all_texts = {}
	all_directories = {}
	all_durations = {}
	speaker_audio_dict = {}
	speaker_list = []

	utt2spk_dict = {}

	if lang == 'swahili':
		with io.open(utt2spk_file, encoding = 'utf-8') as f:
			for line in f:
				toks = line.strip().split()
				utt2spk_dict[toks[0]] = toks[1]

	for i in range(len(audio_info_data)):
		audio = audio_info_data[i]
		audio_time = float(audio[-2])
		all_durations[audio[0]] = audio_time
		time += audio_time
		all_audios.append(audio[0])
		all_directories[audio[0]] = audio[1]
		all_texts[audio[0]] = audio[-1]
	
		speaker = audio[0].split('_')[1]
	
		if lang == 'iban':
			speaker = '_'.join(w for w in audio[0].split('_')[ : 2])
		if lang == 'fongbe':
			speaker = audio[0].split('_')[0]
		if lang == 'swahili':
			speaker = utt2spk_dict[audio[0]]
	
		speaker_list.append(speaker)

		if speaker not in speaker_audio_dict:
			speaker_audio_dict[speaker] = [audio[0]]
		else:
			speaker_audio_dict[speaker].append(audio[0])

	for speaker in set(speaker_list):
		train_audio = []
		dev_audio = []
		train_texts = []
		dev_texts = []
		train_wav = []
		dev_wav = []

		for k, v in speaker_audio_dict.items():
			if k != speaker:
				for audio in v:
					train_audio.append(audio)
			if k == speaker:
				dev_audio = v

		new_train_audio = ''
		new_dev_audio = ''

		if lang == 'iban':
			new_train_audio = iban_sort_audio_id(train_audio)
			new_dev_audio = iban_sort_audio_id(dev_audio)

		if lang == 'fongbe':
			new_train_audio = fongbe_sort_audio_id(train_audio)
			new_dev_audio = fongbe_sort_audio_id(dev_audio)

		if lang == 'swahili':
			new_train_audio = swahili_sort_audio_id(train_audio, utt2spk_file)
			new_dev_audio = swahili_sort_audio_id(dev_audio, utt2spk_file)

		if lang == 'wolof': 
			new_train_audio = sort_audio_id(train_audio)
			new_dev_audio = sort_audio_id(dev_audio)

		for i in range(len(new_train_audio)):
			audio = new_train_audio[i]
			if audio + ' ' + all_texts[audio] not in train_texts:
				train_texts.append(audio + ' ' + all_texts[audio])
			if audio + ' ' + all_directories[audio] not in train_wav:
				train_wav.append(audio + ' ' + all_directories[audio])

		for i in range(len(new_dev_audio)):
			audio = new_dev_audio[i]
			if audio + ' ' + all_texts[audio] not in dev_texts:
				dev_texts.append(audio + ' ' + all_texts[audio])
			if audio + ' ' + all_directories[audio] not in dev_wav:
				dev_wav.append(audio + ' ' + all_directories[audio])

		split_id = ''

		if speaker.startswith('000'):
			split_id = speaker[3:]
		elif speaker.startswith('00'):
			split_id = speaker[2:]
		elif speaker.startswith('0'):
			split_id = speaker[1:]
		else:
			split_id = speaker

		if not os.path.exists(output + 'train' + split_id):
			os.makedirs(output + 'train' + split_id)
		if not os.path.exists(output + 'dev' + split_id):
			os.makedirs(output + 'dev' + split_id)

		write_text_wav(train_texts, train_wav, dev_texts, dev_wav, split_id, output)

		write_utt_spk(output, split_id, 'heldout', args.speaker, args.lang)
		text_to_corpus(output, split_id)

	return new_train_audio, new_dev_audio


### Random splits, performing speaker normalization separately ###
### for audios of different speakers ###

def random_split(audio_info_data, output, split_id, lang, utt2spk_file = None):

	time = 0
	all_audios = []
	all_texts = {}
	all_directories = {}
	all_durations = {}

	for i in range(len(audio_info_data)):
		audio = audio_info_data[i]
		audio_time = float(audio[-2])
		all_durations[audio[0]] = audio_time
		time += audio_time
		all_audios.append(audio[0])
		all_directories[audio[0]] = audio[1]
		all_texts[audio[0]] = audio[-1]

	train_time = time * (1 - HELDOUT_RATE)
	dev_time = time - train_time

	print(time / (60 * 60))
	print(train_time / (60 * 60))
	print(dev_time / (60 * 60))

	train_audio = []
	dev_audio = []

	index_list = []
	for i in range(len(all_audios)):
		index_list.append(i)

	random.shuffle(index_list)

	start = 0

	dev_index_list = []

	for i in range(len(index_list)):
		while start <= dev_time:
			idx = index_list[i]
			audio = all_audios[idx]
			dev_audio.append(audio)
			start += all_durations[audio]
			dev_index_list.append(idx)
			index_list.remove(idx)

	for idx in index_list:
		if idx not in dev_index_list:
			audio = all_audios[idx]
			train_audio.append(audio)

	new_train_audio = ''
	new_dev_audio = ''

	if lang == 'iban':
		new_train_audio = iban_sort_audio_id(train_audio)
		new_dev_audio = iban_sort_audio_id(dev_audio)

	if lang == 'fongbe':
		new_train_audio = fongbe_sort_audio_id(train_audio)
		new_dev_audio = fongbe_sort_audio_id(dev_audio)

	if lang in ['swahili', 'bemba']:
		new_train_audio = swahili_sort_audio_id(train_audio, utt2spk_file)
		new_dev_audio = swahili_sort_audio_id(dev_audio, utt2spk_file)

	if lang == 'wolof':
		new_train_audio = sort_audio_id(train_audio)
		new_dev_audio = sort_audio_id(dev_audio)

	train_texts = []
	dev_texts = []

	train_wav = []
	dev_wav = []

	for i in range(len(new_train_audio)):
		audio = new_train_audio[i]
		if audio + ' ' + all_texts[audio] not in train_texts:
			train_texts.append(audio + ' ' + all_texts[audio])
		if audio + ' ' + all_directories[audio] not in train_wav:
			train_wav.append(audio + ' ' + all_directories[audio])

	for i in range(len(new_dev_audio)):
		audio = new_dev_audio[i]
		if audio + ' ' + all_texts[audio] not in dev_texts:
			dev_texts.append(audio + ' ' + all_texts[audio])
		if audio + ' ' + all_directories[audio] not in dev_wav:
			dev_wav.append(audio + ' ' + all_directories[audio])

	write_text_wav(train_texts, train_wav, dev_texts, dev_wav, split_id, output)

	return new_train_audio, new_dev_audio



def write_utt_spk(output, split_id, split, speaker, lang):

	with io.open(output + lang + '_' + split + '_' + speaker + '_utt_spk' + split_id + '.sh', 'w', encoding = 'utf-8') as f:
		f.write("echo 'make utt2spk and spk2utt for train dev...'" + '\n')
		f.write("for dir in " + output + 'train' + split_id + ' ' + output + 'dev' + split_id + '\n')
		f.write('do' + '\n')
		f.write("	cat $dir/text | cut -d' ' -f1 > $dir/utt" + '\n')
		f.write("	cat $dir/text | cut -d'_' -f2 > $dir/spk" + '\n')
		f.write('	paste $dir/utt $dir/spk > $dir/utt2spk' + '\n')
		f.write('	utils/utt2spk_to_spk2utt.pl $dir/utt2spk | sort -k1 > $dir/spk2utt' + '\n')
		f.write('	rm $dir/utt $dir/spk' + '\n')
		f.write('done' + '\n')

	if lang == 'bemba':
		with io.open(output + lang + '_' + split + '_' + speaker + '_utt_spk' + split_id + '.sh', 'w', encoding = 'utf-8') as f:
			f.write("echo 'make utt2spk and spk2utt for train dev...'" + '\n')
			f.write("for dir in " + output + 'train' + split_id + ' ' + output + 'dev' + split_id + '\n')
			f.write('do' + '\n')
			f.write("	cat $dir/text | cut -d' ' -f1 > $dir/utt" + '\n')
			f.write("	cat $dir/text | cut -d'_' -f1 > $dir/id1" + '\n')
			f.write("	cat $dir/text | cut -d'_' -f2 > $dir/id2" + '\n')
			f.write("	paste -d '_' $dir/id1 $dir/id2 > $dir/spk" + '\n')
			f.write('	paste $dir/utt $dir/spk > $dir/utt2spk' + '\n')
			f.write('	utils/utt2spk_to_spk2utt.pl $dir/utt2spk | sort -k1 > $dir/spk2utt' + '\n')
			f.write('	rm $dir/utt $dir/spk' + '\n')
			f.write('done' + '\n')

	if lang == 'iban':
		with io.open(output + lang + '_' + split + '_' + speaker + '_utt_spk' + split_id + '.sh', 'w', encoding = 'utf-8') as f:
			f.write("echo 'make utt2spk and spk2utt for train dev...'" + '\n')
			f.write("for dir in " + output + 'train' + split_id + ' ' + output + 'dev' + split_id + '\n')
			f.write('do' + '\n')
			f.write("	cat $dir/text | cut -d' ' -f1 > $dir/utt" + '\n')
			f.write("	cat $dir/text | cut -d'_' -f1 > $dir/gender" + '\n')
			f.write("	cat $dir/text | cut -d'_' -f2 > $dir/id" + '\n')
			f.write("	paste -d '_' $dir/gender $dir/id > $dir/spk" + '\n')
			f.write('	paste $dir/utt $dir/spk > $dir/utt2spk' + '\n')
			f.write('	utils/utt2spk_to_spk2utt.pl $dir/utt2spk | sort -k1 > $dir/spk2utt' + '\n')
			f.write('	rm $dir/utt $dir/spk' + '\n')
			f.write('done' + '\n')

	if lang in ['fongbe', 'swahili']:
		with io.open(output + lang + '_' + split + '_' + speaker + '_utt_spk' + split_id + '.sh', 'w', encoding = 'utf-8') as f:
			f.write("echo 'make utt2spk and spk2utt for train dev...'" + '\n')
			f.write("for dir in " + output + 'train' + split_id + ' ' + output + 'dev' + split_id + '\n')
			f.write('do' + '\n')
			f.write("	cat $dir/text | cut -d' ' -f1 > $dir/utt" + '\n')
			f.write("	cat $dir/text | cut -d'_' -f1 > $dir/spk" + '\n')
			f.write('	paste $dir/utt $dir/spk > $dir/utt2spk' + '\n')
			f.write('	utils/utt2spk_to_spk2utt.pl $dir/utt2spk | sort -k1 > $dir/spk2utt' + '\n')
			f.write('	rm $dir/utt $dir/spk' + '\n')
			f.write('done' + '\n')


	with io.open(output + lang + '_' + split + '_' + speaker + '_compute_mfcc' + split_id + '.sh', 'w', encoding = 'utf-8') as f:
		f.write("echo 'compute mfcc for train dev...'" + '\n')
		f.write("for dir in " + output + 'train' + split_id + ' ' + output + 'dev' + split_id + '\n')
		f.write('do' + '\n')
		f.write("	steps/make_mfcc.sh --nj 4 $dir exp/make_mfcc/$dir $dir/mfcc" + '\n')
		f.write("	steps/compute_cmvn_stats.sh $dir exp/make_mfcc/$dir $dir/mfcc" + '\n')
		f.write("done" + '\n')


### Usage
### python3 scripts/2.split_data.py --output data_lexicon/fongbe/ --lang fongbe --utt2spk data_lexicon/fongbe/all_utt2spk

### For simulation experiments
#python3 scripts/2.split_data.py --output data_lexicon/bemba/top_tier/1/ --lang bemba --utt2spk data_lexicon/bemba/all_utt2spk --start 1 --n 2
#python3 scripts/2.split_data.py --output data_lexicon/bemba/top_tier/2/ --lang bemba --utt2spk data_lexicon/bemba/all_utt2spk --start 2 --n 3

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
#	parser.add_argument('--info', type = str, help = 'path + file of audio_info.txt')
	parser.add_argument('--output', type = str, help = 'output path')
	parser.add_argument('--split', default = 'random', help = 'split method')
	parser.add_argument('--speaker', default = 'same', help = 'whether perform speaker normalization for each speaker seaparately')
	parser.add_argument('--lang', type = str, help = 'language')
	parser.add_argument('--utt2spk', type = str, help = 'utt2spk file')
	parser.add_argument('--n', default = '4', help = 'how many random splits')
	parser.add_argument('--start', default = '1', help = 'start number of the random split')
	args = parser.parse_args()

	print(args.lang)
	
	audio_info_data = gather_audio_info(args.output + 'audio_info.txt')

	functions = {'random': random_split}

	if args.split == 'random':
		if not os.path.exists(args.output + 'random'):
			os.makedirs(args.output + 'random')

		for i in range(int(args.start), int(args.n)):
			i = str(i)

			if not os.path.exists(args.output + 'random/train' + i):
				os.makedirs(args.output + 'random/train' + i)

			if not os.path.exists(args.output + 'random/dev' + i):
				os.makedirs(args.output + 'random/dev' + i)

			new_train_audio, new_dev_audio = 0, 0

			if args.lang not in ['swahili', 'bemba']:
				new_train_audio, new_dev_audio = random_split(audio_info_data, args.output + 'random/', i, args.lang)
			else:
				new_train_audio, new_dev_audio = random_split(audio_info_data, args.output + 'random/', i, args.lang, args.utt2spk)

			write_utt_spk(args.output + 'random/', i, args.split, args.speaker, args.lang)

			text_to_corpus(args.output + 'random/', i)


