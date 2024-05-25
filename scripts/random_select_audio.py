import io, os, argparse, random
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--path', type = str, help = 'path to audio_info.txt')
args = parser.parse_args()

top_tier_total = 5700
second_tier_total = 27420

audio_info = pd.read_csv(args.path + 'audio_info.txt', '\t')
files = audio_info['File'].tolist()
paths = audio_info['Path'].tolist()
durations = audio_info['Duration'].tolist()
transcripts = audio_info['Transcript'].tolist()
audio_info_dict = {}
for i in range(len(files)):
	file = files[i]
	path = paths[i]
	duration = durations[i]
	transcript = transcripts[i]
	audio_info_dict[file] = [path, duration, transcript]

total_duration = sum([float(duration) for duration in durations])
print(total_duration)

header = ['File', 'Path', 'Duration', 'Transcript']

for n in range(2, 6): ### how many augmented data sets to construct

	if total_duration > top_tier_total:
		temp_audio_info_dict = audio_info_dict

		top_tier_select_files = []
		top_tier_select_duration = 0
	
		if not os.path.exists(args.path + 'top_tier/'):
			os.system('mkdir ' + args.path + 'top_tier/')

		if not os.path.exists(args.path + 'top_tier/' + str(n)):
			os.system('mkdir ' + args.path + 'top_tier/' + str(n))

		random.shuffle(files)
		i = 0
		while top_tier_select_duration <= top_tier_total + 2:
			file = files[i]
			try:
				top_tier_select_duration += float(temp_audio_info_dict[file][1])
				top_tier_select_files.append(file)
			except:
				top_tier_select_duration += float(temp_audio_info_dict[file][2])
				top_tier_select_files.append(file)

			i += 1

			if top_tier_select_duration > top_tier_total + 2:
				break
		
		print(top_tier_select_duration)
		print('\n')
		
		with open(args.path + 'top_tier/' + str(n) + '/audio_info.txt', 'w') as f:
			f.write('\t'.join(w for w in header) + '\n')
			for file in top_tier_select_files:
				info = audio_info_dict[file]
				if info[0] == file:
					f.write('\t'.join(str(w) for w in info) + '\n')
				else:
					info.insert(0, file)
					f.write('\t'.join(str(w) for w in info) + '\n')

	if total_duration > second_tier_total:
		temp_audio_info_dict = audio_info_dict

		second_tier_select_files = []
		second_tier_select_duration = 0
	
		if not os.path.exists(args.path + 'second_tier/'):
			os.system('mkdir ' + args.path + 'second_tier/')

		if not os.path.exists(args.path + 'second_tier/' + str(n)):
			os.system('mkdir ' + args.path + 'second_tier/' + str(n))

		random.shuffle(files)
		i = 0
		while second_tier_select_duration <= second_tier_total + 2:
			file = files[i]
			try:
				second_tier_select_duration += float(temp_audio_info_dict[file][1])
				second_tier_select_files.append(file)
			except:
				second_tier_select_duration += float(temp_audio_info_dict[file][2])
				second_tier_select_files.append(file)

			i += 1

			if second_tier_select_duration > second_tier_total + 2:
				break
	
		print(second_tier_select_duration)

		with open(args.path + 'second_tier/' + str(n) + '/audio_info.txt', 'w') as f:
			f.write('\t'.join(w for w in header) + '\n')
			for file in second_tier_select_files:
				info = audio_info_dict[file]
				if info[0] == file:
					f.write('\t'.join(str(w) for w in info) + '\n')
				else:
					info.insert(0, file)
					f.write('\t'.join(str(w) for w in info) + '\n')



