### Randomly select texts to build language models for simulation experiments
### python3 --path data/fongbe/top_tier/ --corpus data/fongbe/local/corpus.txt

import io, os, argparse, random
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--path', type = str, help = 'e.g., data/fongbe/top_tier')
parser.add_argument('--corpus', type = str, help = 'e.g., data/fongbe/local/corpus.txt')
args = parser.parse_args()


proportion = 1.3
train_data_num_of_words = 0
with open(args.path + '/1/random/train1/corpus', encoding = 'latin-1') as f:
	for line in f:
		toks = line.split()
		train_data_num_of_words += len(toks)

total_num_of_words = int(proportion * train_data_num_of_words) + 1
print(total_num_of_words)

transcripts = []
with open(args.corpus, encoding = 'latin-1') as f:
	for line in f:
		toks = line.strip()
		transcripts.append(toks)

for n in range(1, 4): ### how many texts for language model to construct
	select_texts = []
	select_num_of_words = 0

	random.shuffle(transcripts)
	i = 0
	while select_num_of_words < total_num_of_words:
		transcript = transcripts[i]
		select_num_of_words += len(transcript.split())
		select_texts.append(transcript)
		i += 1

		if select_num_of_words >= total_num_of_words:
			break

	print(select_num_of_words)

	proportion = str(proportion)[ : 3].replace('.', '')
	with open(args.path + 'proportion_corpus.' + proportion + '.' + str(n), 'w') as f:
		for transcript in select_texts:
			f.write(transcript + '\n')
