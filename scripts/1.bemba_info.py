import io, os, argparse
import pandas as pd

def get_data(file):
	data = pd.read_csv(file, sep = '\t')
	try:
		data['audio']
	except:
		data = pd.read_csv(file)
	data_audio = data['audio'].tolist()
	data_path = []
	for i in range(len(data_audio)):
		path = '/data/liuaal/asr_lm_size/BembaSpeech/data/audio/' + data_audio[i]
		data_path.append(path)
	data['Path'] = data_path
	audio_map = {}
	new_data_audio = []
	data_idx = []
#	start = 0
	for i in range(len(data_audio)):
		audio = data_audio[i].split('.')[0]
		audio = audio.split('_')
		new_audio = audio[1 : ]
		data_idx.append('_'.join(w for w in new_audio[ : 2]))
		new_audio.append(audio[0])
	#	new_audio.append(str(start))
		new_audio = '_'.join(w for w in new_audio)
		new_data_audio.append(new_audio)
	#	start += 1
		audio_map['_'.join(w for w in audio)] = new_audio
	data['audio'] = new_data_audio
	data = data.rename(columns={"audio": "File", "duration": "Duration", "sentence": "Transcript"}, errors="raise")
	return data, data_idx, audio_map

## audio_info.txt
#header = ['File', 'Path', 'Duration', 'Transcript']

train_data, train_idx, train_map = get_data('BembaSpeech/data/splits/train.csv')
dev_data, dev_idx, dev_map = get_data('BembaSpeech/data/splits/dev.csv')
test_data, test_idx, test_map = get_data('BembaSpeech/data/splits/test.csv')

data = pd.concat([train_data, dev_data, test_data], axis = 0)
if not os.path.exists('data/bemba'):
	os.system('mkdir data/bemba')

audio_info_file = data[['File', 'Path', 'Duration', 'Transcript']]
audio_info_file.to_csv('data/bemba/audio_info.txt', sep = '\t', index = False)

full_audio_map = train_map
full_audio_map.update(dev_map)
full_audio_map.update(test_map)

import json
with open('data/bemba/full_audio_map.json', 'w') as fp:
    json.dump(full_audio_map, fp)


### all_utt2spk

all_utt2spk = []
all_utt2spk_dict = {}

train_audio = train_data['File']
for i in range(len(train_audio)):
	audio = train_audio[i]
	idx = train_idx[i]
	all_utt2spk.append(audio + ' ' + idx)
	all_utt2spk_dict[audio] = idx

dev_audio = dev_data['File']
for i in range(len(dev_audio)):
	audio = dev_audio[i]
	idx = dev_idx[i]
	all_utt2spk.append(audio + ' ' + idx)
	all_utt2spk_dict[audio] = idx

test_audio = test_data['File']
for i in range(len(test_audio)):
	audio = test_audio[i]
	idx = test_idx[i]
	all_utt2spk.append(audio + ' ' + idx)
	all_utt2spk_dict[audio] = idx

with io.open('data/bemba/all_utt2spk', 'w') as f:
	for tok in all_utt2spk:
		f.write(tok + '\n')

wav_dict = {}
text_dict = {}
train_wav = {}
train_text = {}
train_utt2spk = {}
test_wav = {}
test_text = {}
test_utt2spk = {}

audio_info = pd.read_csv('data/bemba/audio_info.txt', sep = '\t')
ids = audio_info['File']
transcripts = audio_info['Transcript']
paths = audio_info['Path']
for i in range(len(ids)):
	idx = ids[i]
	wav_dict[idx] = paths[i]
	text_dict[idx] = transcripts[i]

train = pd.read_csv('BembaSpeech/data/splits/train.csv')
test = pd.read_csv('BembaSpeech/data/splits/test.csv', sep = '\t')

for i in range(len(train)):
	temp_idx = train['audio'].tolist()[i].split('.')[0]
	idx = full_audio_map[temp_idx]
	wav = wav_dict[idx]
	text = text_dict[idx]
	utt2spk = all_utt2spk_dict[idx]
	train_wav[idx] = wav
	train_text[idx] = text
	train_utt2spk[idx] = utt2spk

for i in range(len(test)):
	temp_idx = test['audio'].tolist()[i].split('.')[0]
	idx = full_audio_map[temp_idx]
	wav = wav_dict[idx]
	text = text_dict[idx]
	utt2spk = all_utt2spk_dict[idx]
	test_wav[idx] = wav
	test_text[idx] = text
	test_utt2spk[idx] = utt2spk

with open('data/bemba/original/train1/text', 'w') as f:
	for k, v in train_text.items():
		f.write(k + ' ' + v + '\n')

with open('data/bemba/original/train1/wav.scp', 'w') as f:
	for k, v in train_wav.items():
		f.write(k + ' ' + v + '\n')

with open('data/bemba/original/train1/utt2spk', 'w') as f:
	for k, v in train_utt2spk.items():
		f.write(k + ' ' + v + '\n')

with open('data/bemba/original/dev1/text', 'w') as f:
	for k, v in test_text.items():
		f.write(k + ' ' + v + '\n')

with open('data/bemba/original/dev1/wav.scp', 'w') as f:
	for k, v in test_wav.items():
		f.write(k + ' ' + v + '\n')

with open('data/bemba/original/dev1/utt2spk', 'w') as f:
	for k, v in test_utt2spk.items():
		f.write(k + ' ' + v + '\n')


### Reordering utt2spk
all_utt2spk = {}
audio_ids = []
max_mid_num = 0
max_num = 0
with open('data/bemba/original/train1/utt2spk') as f:
	for line in f:
		toks = line.strip().split()
		idx = toks[0].split('_elicit_')[0]
		if int(toks[0].split('_elicit_')[1].split('_')[0]) > max_mid_num:
			max_mid_num = int(toks[0].split('_elicit_')[1].split('_')[0])
		if int(toks[0].split('_elicit_')[1].split('_')[1]) > max_num:
			max_num = int(toks[0].split('_elicit_')[1].split('_')[1])
		idx = idx # + 'elicit'
		if idx not in audio_ids:
			audio_ids.append(idx)
		all_utt2spk[toks[0]] = toks[1]

new_all_utt2spk = []
for idx in audio_ids:
	for i in range(0, max_num + 1):
		for z in range(0, max_mid_num + 1):		
			if idx + '_elicit_' + str(z) + '_' + str(i) in all_utt2spk:
				new_idx = idx + '_elicit_' + str(z) + '_' + str(i)
				info =  new_idx + ' ' + all_utt2spk[new_idx]
				new_all_utt2spk.append(info)


with open('data/bemba/original/train1/utt2spk', 'w') as f:
	for tok in new_all_utt2spk:
		f.write(tok + '\n')


all_utt2spk = {}
audio_ids = []
with open('data/bemba/original/dev1/utt2spk') as f:
	for line in f:
		toks = line.strip().split()
		idx = toks[0].split('elicit')[0]
		idx = idx + 'elicit'
		if idx not in audio_ids:
			audio_ids.append(idx)
		all_utt2spk[toks[0]] = toks[1]

new_all_utt2spk = []
for idx in audio_ids:
	for i in range(0, 51):
		if idx + '_' + str(i) in all_utt2spk:
			new_idx = idx + '_' + str(i)
			info =  new_idx + ' ' + all_utt2spk[new_idx]
			new_all_utt2spk.append(info)

with open('data/bemba/original/dev1/utt2spk', 'w') as f:
	for tok in new_all_utt2spk:
		f.write(tok + '\n')



