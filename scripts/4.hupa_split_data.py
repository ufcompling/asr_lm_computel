import io, os, argparse, random, librosa, statistics, glob
from pydub import AudioSegment
import parselmouth
import numpy as np
import seaborn as sns

from collections import Counter
from scipy.stats import wasserstein_distance

import collections
from typing import Dict, Generator, Iterator, List, Set, Text, Tuple

import pandas as pd
from scipy import stats
from sklearn import feature_extraction
from sklearn import neighbors

HELDOUT_RATE = 0.2

def gather_audio_info(file):

	data = []

	if 'lm' not in file:
		with io.open(file, encoding = 'utf-8') as f:
			for line in f:
				toks = line.strip().split('\t')
				if toks[0].startswith('File') is False or toks[0] != 'File':
					new_toks = []
					for w in toks[ : -1]:
						new_toks.append(w)
					new_transcript = []
					for w in toks[-1].split()[:-2]:
						new_transcript.append(w)
					new_transcript = ' '.join(w for w in new_transcript)
					new_toks.append(new_transcript)
					data.append(new_toks)
	else:
		with io.open(file, encoding = 'utf-8') as f:
			for line in f:
				toks = line.strip().split('\t')
				if toks[0].startswith('File') is False or toks[0] != 'File':
					new_toks = []
					for w in toks[ : 3]:
						new_toks.append(w)
					new_transcript = []
					for w in toks[3].split()[:-2]:
						new_transcript.append(w)
					new_transcript = ' '.join(w for w in new_transcript)
					new_toks.append(new_transcript)
					for w in toks[-3 : ]:
						new_toks.append(w)
					data.append(new_toks)

	return data

def sort(data, num_of_speakers):

	new_data = []
	text_ids = []

	for idx, text in data.items():
		while idx[0] == '0':
			idx = idx[1 :]
		text_ids.append(int(idx))

	text_ids.sort()

	for i in range(len(text_ids)):
		idx = text_ids[i]
		new_idx = ''		
		if len(str(idx)) == 1:
			new_idx = '000' + str(idx)
		if len(str(idx)) == 2:
			new_idx = '00' + str(idx)
		if len(str(idx)) == 3:
			new_idx = '0' + str(idx)
		if len(str(idx)) == 4:
			new_idx = str(idx)
		
		new_data.append('verdena_' + num_of_speakers + '_' + new_idx + ' ' + ' '.join(w for w in data[str(idx)]))

	return new_data

### Get corpus from training data ###

def text_to_corpus(text):

	corpus = []

	with io.open(text, encoding = 'utf-8') as f:
		for line in f:
			toks = line.strip().split()
			transcripts = toks[1 : ]
			transcripts = ' '.join(w for w in transcripts)
			corpus.append(transcripts)

	return corpus

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

	sub_dir = output.split('/')[-2] + '/'

	with io.open(output + lang + '_' + split + '_' + speaker + '_compute_mfcc' + split_id + '.sh', 'w', encoding = 'utf-8') as f:
		f.write("echo 'compute mfcc for train dev...'" + '\n')
		f.write("for dir in " + output + 'train' + split_id + ' ' + output + 'dev' + split_id + '\n')
		f.write('do' + '\n')
		f.write("	steps/make_mfcc.sh --nj 4 $dir exp/make_mfcc/$dir $dir/mfcc" + '\n')
		f.write("	steps/compute_cmvn_stats.sh $dir exp/make_mfcc/$dir $dir/mfcc" + '\n')
		f.write("done" + '\n')

### Random splits ###

def random_split(file, num_of_speakers, audio_info_data, output, split_id):

	all_audios = []
	all_texts = {}
	all_directories = {}

	for i in range(len(audio_info_data)):
		audio = audio_info_data[i]
		audio_time = float(audio[-2])
		all_audios.append(audio[0])
		all_directories[audio[0]] = audio[1]
		all_texts[audio[0]] = audio[-1]

	data = []
	time_list = []

	with io.open(file, encoding = 'utf-8') as f:
		for line in f:
			toks = line.strip().split()
			data.append(toks)
			time_list.append(float(toks[-1]) - float(toks[-2]))

	time = sum(time_list)
	train_time = time * (1 - HELDOUT_RATE)
	dev_time = time - train_time

	train_data = {}
	dev_data = {}

	index_list = []
	for i in range(len(data)):
		index_list.append(i)

	random.shuffle(index_list)

	start = 0
	c = 0

	for i in range(len(index_list)):
		while start <= dev_time:
			tok = data[index_list[i]][0].split('_')
			idx = tok[-1]
			while idx[0] == '0':
				idx = idx[1 :]
			if idx in dev_data:
				print(dev_data[idx])
			dev_data[idx] = data[index_list[i]][1 : -2]
			start += time_list[index_list[i]]
			index_list.remove(index_list[i])
			c += 1

	for i in index_list:
		tok = data[i][0].split('_')
		idx = tok[-1]
		while idx[0] == '0':
			idx = idx[1 :]
		if idx in train_data:
			print(train_data[idx])
		train_data[idx] = data[i][1 : -2]

	new_train_audio = sort(train_data, num_of_speakers)
	new_dev_audio = sort(dev_data, num_of_speakers)

	train_texts = []
	dev_texts = []

	train_wav = []
	dev_wav = []

	for i in range(len(new_train_audio)):
		audio = new_train_audio[i].split()[0]
		train_texts.append(audio + ' ' + all_texts[audio])
		train_wav.append(audio + ' ' + all_directories[audio])

	for i in range(len(new_dev_audio)):
		audio = new_dev_audio[i].split()[0]
		dev_texts.append(audio + ' ' + all_texts[audio])
		dev_wav.append(audio + ' ' + all_directories[audio])

	write_text_wav(train_texts, train_wav, dev_texts, dev_wav, split_id, output)

	return new_train_audio, new_dev_audio 

#	return sort(train_data, num_of_speakers), sort(dev_data, num_of_speakers)

### Usage
### python3 scripts/4.hupa_split_data.py --input ../kaldi/data/hupa/top_tier/ --output data_lexicon/hupa/top_tier/ --n 1

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type = str, help = 'input path; e.g., ../kaldi/data/hupa/top_tier')
	parser.add_argument('--n', type = str, help = '1 (top tier) or 2(top tier)')
	parser.add_argument('--split', default = 'random', help = 'split method')
#	parser.add_argument('--info', type = str, help = 'path + file of audio_info.txt')
	parser.add_argument('--lang', default = 'hupa', help = 'language')
	parser.add_argument('--speaker', default = 'same', help = 'different speakers or treating as the same')
	parser.add_argument('--output', type = str, help = 'output path; e.g., data_lexicon/hupa/top_tier')
	args = parser.parse_args()

	temp_audio_info_data = gather_audio_info('data_lexicon/hupa/audio_info.txt')
	audio_info_data = []
	for tok in temp_audio_info_data:
		file = tok[0]
		file = file.split('_')
		if file[1] == args.n:
			audio_info_data.append(tok)

	quality = ''

	if args.n == '1':
		quality = 'top_tier'

	if args.n == '2':
		quality = 'second_tier'

	if not os.path.exists('data_lexicon/hupa/' + quality + '/'):
		os.makedirs('data_lexicon/hupa/' + quality + '/')

	### If doing random splits ###

	if args.split == 'random':

		if not os.path.exists('data_lexicon/hupa/' + quality + '/random/'):
			os.makedirs('data_lexicon/hupa/' + quality + '/random/')

		for i in range(1, 4):

			i = str(i)

			if not os.path.exists('data_lexicon/hupa/' + quality + '/random/train' + str(i)):
				os.makedirs('data_lexicon/hupa/' + quality + '/random/train' + str(i))

			if not os.path.exists('data_lexicon/hupa/' + quality + '/random/dev' + str(i)):
				os.makedirs('data_lexicon/hupa/' + quality + '/random/dev' + str(i))

			if not os.path.exists('data_lexicon/hupa/' + quality + '/random/train' + str(i) + '/' + str(args.n) + '/'):
				os.makedirs('data_lexicon/hupa/' + quality + '/random/train' + str(i) + '/' + str(args.n) + '/')

			if not os.path.exists('data_lexicon/hupa/' + quality + '/random/dev' + str(i) + '/' + str(args.n) + '/'):
				os.makedirs('data_lexicon/hupa/' + quality + '/random/dev' + str(i) + '/' + str(args.n) + '/')


			train_data, dev_data = random_split(args.input + 'all_text' + args.n + '.txt', args.n, audio_info_data, args.output + '/random/', str(i))

		#	train_f = ''

		#	with io.open('data_lexicon/hupa/' + quality + '/random/train' + str(i) + '/text.' + args.n, 'w') as train_f:
		#		for tok in train_data:	
		#			train_f.write(tok + '\n')

		#	os.system('cat data_lexicon/hupa/' + quality + '/random/train' + str(i) + '/text.* > data_lexicon/hupa/' + quality + '/random/train' + str(i) + '/temp')
		#	os.system('mv data_lexicon/hupa/' + quality + '/random/train' + str(i) + '/temp data_lexicon/hupa/' + quality + '/random/train' + str(i) + '/text')
			
			corpus = text_to_corpus('data_lexicon/hupa/' + quality + '/random/train' + str(i) + '/text')
		
			with io.open('data_lexicon/hupa/' + quality + '/random/train' + str(i) + '/corpus', 'w', encoding = 'utf-8') as f:
				for tok in corpus:
					f.write(tok + '\n')

		#	dev_f = ''

		#	with io.open('data_lexicon/hupa/' + quality + '/random/dev' + str(i) + '/text.' + args.n, 'w') as dev_f:
		#		for tok in dev_data:
		#			dev_f.write(tok + '\n')

		#	os.system('cat data_lexicon/hupa/' + quality + '/random/dev' + str(i) + '/text.* > data_lexicon/hupa/' + quality + '/random/dev' + str(i) + '/temp')
		#	os.system('mv data_lexicon/hupa/' + quality + '/random/dev' + str(i) + '/temp data_lexicon/hupa/' + quality + '/random/dev' + str(i) + '/text')
			
			with io.open('random_sort_data' + str(i) + '.sh', 'w') as f:
				for tok in train_data:
					file_name = tok.split()[0]
					audio_file = file_name + '.wav'

					f.write('cp ' + args.input + audio_file + ' ' + 'data_lexicon/hupa/' + quality + '/random/train' + str(i) + '/' + str(args.n) + '/' + '\n')

				for tok in dev_data:
					file_name = tok.split()[0]
					audio_file = file_name + '.wav'

					f.write('cp ' + args.input + audio_file + ' ' + 'data_lexicon/hupa/' + quality + '/random/dev' + str(i) + '/' + str(args.n) + '/' + '\n')

			os.system('bash random_sort_data' + str(i) + '.sh')

			write_utt_spk('data_lexicon/hupa/' + quality + '/' + args.split + '/', i, args.split, args.speaker, args.lang)

