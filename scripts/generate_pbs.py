import sys, io, os


for lg in ['fongbe', 'wolof', 'iban', 'swahili', 'bemba']:
	for n in range(1, 6):
		for size in ['base', 'large']:
			with open(lg + '_' + size + str(n) + '.pbs', 'w') as f:
				f.write("#!/bin/bash" + '\n')
				f.write("#SBATCH --job-name='" + lg + " random'" + '\n')
				f.write("#SBATCH --ntasks 1 --cpus-per-task 4" + '\n')
				f.write("#SBATCH --mem=55gb" + '\n')
				f.write("#SBATCH --time=48:00:00" + '\n')
				f.write("#SBATCH --mail-type=BEGIN,END,FAIL." + '\n')
				f.write("#SBATCH --partition=gpuv100" + '\n')
				f.write("module load cuda10.2" + '\n')
				f.write("module load kaldi" + '\n')
				f.write("module load cudnn7.6-cuda10.2" + '\n')
				f.write('\n')
				f.write('cd /data/liuaal/asr_lm_size/' + '\n')
				f.write('\n')
				f.write('bash cmds/random_lexicon.sh ' + lg + ' ' + size + ' ' + str(n) + '\n')
				f.write('\n')

for lg in ['fongbe', 'wolof', 'iban', 'swahili', 'bemba']:
	for n in range(1, 6):
		for proportion in ['16']:
			size = 'proportion' + proportion + '.' + str(n)
			with open(lg + '_' + size + '.pbs', 'w') as f:
				f.write("#!/bin/bash" + '\n')
				f.write("#SBATCH --job-name='" + lg + " random'" + '\n')
				f.write("#SBATCH --ntasks 1 --cpus-per-task 4" + '\n')
				f.write("#SBATCH --mem=55gb" + '\n')
				f.write("#SBATCH --time=48:00:00" + '\n')
				f.write("#SBATCH --mail-type=BEGIN,END,FAIL." + '\n')
				f.write("#SBATCH --partition=gpuv100" + '\n')
				f.write("module load cuda10.2" + '\n')
				f.write("module load kaldi" + '\n')
				f.write("module load cudnn7.6-cuda10.2" + '\n')
				f.write('\n')
				f.write('cd /data/liuaal/asr_lm_size/' + '\n')
				f.write('\n')
				f.write('bash cmds/random_lexicon.sh ' + lg + ' ' + size + ' ' + str(n) + '\n')
				f.write('\n')

for lg in ['fongbe', 'wolof', 'iban', 'swahili', 'bemba']:
	for n in range(2, 6):
		for size in ['base', 'large']:
			for quality in ['top', 'second']:
				with open(lg + '_' + quality + '_' + size + str(n) + '.pbs', 'w') as f:
					f.write("#!/bin/bash" + '\n')
					f.write("#SBATCH --job-name='" + lg + " random'" + '\n')
					f.write("#SBATCH --ntasks 1 --cpus-per-task 4" + '\n')
					f.write("#SBATCH --mem=55gb" + '\n')
					f.write("#SBATCH --time=48:00:00" + '\n')
					f.write("#SBATCH --mail-type=BEGIN,END,FAIL." + '\n')
					f.write("#SBATCH --partition=gpuv100" + '\n')
					f.write("module load cuda10.2" + '\n')
					f.write("module load kaldi" + '\n')
					f.write("module load cudnn7.6-cuda10.2" + '\n')
					f.write('\n')
					f.write('cd /data/liuaal/asr_lm_size/' + '\n')
					f.write('\n')
					f.write('bash cmds/random_simulate_hupa_size_lexicon.sh ' + lg + ' ' + quality + ' ' + size + ' ' + str(n) + '\n')
					f.write('\n')

				os.system('sbatch ' + lg + '_' + quality + '_' + size + str(n) + '.pbs')


for lg in ['fongbe', 'wolof', 'iban', 'swahili', 'bemba']:
	for n in range(1, 6):
		for proportion in ['16']:
			size = 'proportion' + proportion + '.' + str(n)
			for quality in ['top', 'second']:
				with open(lg + '_' + quality + '_' + size + '.pbs', 'w') as f:
					f.write("#!/bin/bash" + '\n')
					f.write("#SBATCH --job-name='" + lg + " random'" + '\n')
					f.write("#SBATCH --ntasks 1 --cpus-per-task 4" + '\n')
					f.write("#SBATCH --mem=55gb" + '\n')
					f.write("#SBATCH --time=48:00:00" + '\n')
					f.write("#SBATCH --mail-type=BEGIN,END,FAIL." + '\n')
					f.write("#SBATCH --partition=gpuv100" + '\n')
					f.write("module load cuda10.2" + '\n')
					f.write("module load kaldi" + '\n')
					f.write("module load cudnn7.6-cuda10.2" + '\n')
					f.write('\n')
					f.write('cd /data/liuaal/asr_lm_size/' + '\n')
					f.write('\n')
					f.write('bash cmds/random_simulate_hupa_size_lexicon.sh ' + lg + ' ' + quality + ' ' + size + ' ' + str(n) + '\n')
					f.write('\n')

				os.system('sbatch ' + lg + '_' + quality + '_' + size + '.pbs')


for lg in ['fongbe', 'wolof', 'iban', 'swahili']:
	for n in range(1, 6):
		for merge in ['200', '500', '1000', '1500', '2000', '2500', '3000', '3500', '4000', '4500', '5000']:
			for lm_order in ['5', '10', '15', '20']:
				for size in ['base', 'large']:
					with open('bpe_pbs/' + lg + '_bpe_' + merge + '_' + lm_order + '_' + size + '_' + str(n) + '.pbs', 'w') as f:
						f.write("#!/bin/bash" + '\n')
						f.write("#SBATCH --job-name='" + lg + " random'" + '\n')
						f.write("#SBATCH --ntasks 1 --cpus-per-task 4" + '\n')
						f.write("#SBATCH --mem=55gb" + '\n')
						f.write("#SBATCH --time=48:00:00" + '\n')
						f.write("#SBATCH --mail-type=BEGIN,END,FAIL." + '\n')
						f.write("#SBATCH --partition=gpuv100" + '\n')
						f.write("module load cuda10.2" + '\n')
						f.write("module load kaldi" + '\n')
						f.write("module load cudnn7.6-cuda10.2" + '\n')
						f.write('\n')
						f.write('cd /data/liuaal/asr_lm_size/' + '\n')
						f.write('\n')
						f.write('bash cmds/random_bpe.sh ' + merge + ' ' + lg + ' ' + size + ' ' + str(n) + ' ' + lm_order + '\n')
						f.write('\n')

				for proportion in ['16']:
					size = 'proportion' + proportion + '.' + str(n)
					with open('bpe_pbs/' + lg + '_bpe_' + merge + '_' + lm_order + '_' + size + '.pbs', 'w') as f:
						f.write("#!/bin/bash" + '\n')
						f.write("#SBATCH --job-name='" + lg + " random'" + '\n')
						f.write("#SBATCH --ntasks 1 --cpus-per-task 4" + '\n')
						f.write("#SBATCH --mem=55gb" + '\n')
						f.write("#SBATCH --time=48:00:00" + '\n')
						f.write("#SBATCH --mail-type=BEGIN,END,FAIL." + '\n')
						f.write("#SBATCH --partition=gpuv100" + '\n')
						f.write("module load cuda10.2" + '\n')
						f.write("module load kaldi" + '\n')
						f.write("module load cudnn7.6-cuda10.2" + '\n')
						f.write('\n')
						f.write('cd /data/liuaal/asr_lm_size/' + '\n')
						f.write('\n')
						f.write('bash cmds/random_bpe_proportion.sh ' + merge + ' ' + lg + ' ' + str(n) + ' ' + proportion + ' ' + lm_order + '\n')
						f.write('\n')

				for size in ['base', 'large']:
					for quality in ['top', 'second']:
						with open('bpe_pbs/' + lg + '_' + quality + '_bpe_' + merge + '_' + lm_order + '_' + size + '_' + str(n) + '.pbs', 'w') as f:
							f.write("#!/bin/bash" + '\n')
							f.write("#SBATCH --job-name='" + lg + " random'" + '\n')
							f.write("#SBATCH --ntasks 1 --cpus-per-task 4" + '\n')
							f.write("#SBATCH --mem=55gb" + '\n')
							f.write("#SBATCH --time=48:00:00" + '\n')
							f.write("#SBATCH --mail-type=BEGIN,END,FAIL." + '\n')
							f.write("#SBATCH --partition=gpuv100" + '\n')
							f.write("module load cuda10.2" + '\n')
							f.write("module load kaldi" + '\n')
							f.write("module load cudnn7.6-cuda10.2" + '\n')
							f.write('\n')
							f.write('cd /data/liuaal/asr_lm_size/' + '\n')
							f.write('\n')
							f.write('bash cmds/random_bpe_simulate.sh ' + merge + ' ' + lg + ' ' + size + ' ' + str(n) + ' ' + lm_order + ' ' + quality + '\n')
							f.write('\n')

				for proportion in ['16']:
					size = 'proportion' + proportion + '.' + str(n)
					for quality in ['top', 'second']:
						with open('bpe_pbs/' + lg + '_' + quality + '_bpe_' + merge + '_' + lm_order + '_' + size + '.pbs', 'w') as f:
							f.write("#!/bin/bash" + '\n')
							f.write("#SBATCH --job-name='" + lg + " random'" + '\n')
							f.write("#SBATCH --ntasks 1 --cpus-per-task 4" + '\n')
							f.write("#SBATCH --mem=55gb" + '\n')
							f.write("#SBATCH --time=48:00:00" + '\n')
							f.write("#SBATCH --mail-type=BEGIN,END,FAIL." + '\n')
							f.write("#SBATCH --partition=gpuv100" + '\n')
							f.write("module load cuda10.2" + '\n')
							f.write("module load kaldi" + '\n')
							f.write("module load cudnn7.6-cuda10.2" + '\n')
							f.write('\n')
							f.write('cd /data/liuaal/asr_lm_size/' + '\n')
							f.write('\n')
							f.write('bash cmds/random_bpe_proportion_simulate.sh ' + merge + ' ' + lg + ' ' + str(n) + ' ' + lm_order + ' ' + proportion + ' ' + quality + '\n')
							f.write('\n')


for lg in ['hupa']:
	for n in range(1, 6):
		for size in ['base', 'large']:
			for quality in ['top', 'second']:
				with open(lg + '_' + quality + '_' + size + str(n) + '.pbs', 'w') as f:
					f.write("#!/bin/bash" + '\n')
					f.write("#SBATCH --job-name='" + lg + " random'" + '\n')
					f.write("#SBATCH --ntasks 1 --cpus-per-task 4" + '\n')
					f.write("#SBATCH --mem=55gb" + '\n')
					f.write("#SBATCH --time=48:00:00" + '\n')
					f.write("#SBATCH --mail-type=BEGIN,END,FAIL." + '\n')
					f.write("#SBATCH --partition=gpuv100" + '\n')
					f.write("module load cuda10.2" + '\n')
					f.write("module load kaldi" + '\n')
					f.write("module load cudnn7.6-cuda10.2" + '\n')
					f.write('\n')
					f.write('cd /data/liuaal/asr_lm_size/' + '\n')
					f.write('\n')
					f.write('bash cmds/hupa_random_lexicon.sh ' + quality + ' ' + size + ' ' + str(n) + '\n')
					f.write('\n')


for lg in ['hupa']:
	for n in range(1, 6):
		for proportion in ['16']:
			size = 'proportion' + proportion + '.' + str(n)
			with open(lg + '_top_' + size + '.pbs', 'w') as f:
				f.write("#!/bin/bash" + '\n')
				f.write("#SBATCH --job-name='" + lg + " random'" + '\n')
				f.write("#SBATCH --ntasks 1 --cpus-per-task 4" + '\n')
				f.write("#SBATCH --mem=55gb" + '\n')
				f.write("#SBATCH --time=48:00:00" + '\n')
				f.write("#SBATCH --mail-type=BEGIN,END,FAIL." + '\n')
				f.write("#SBATCH --partition=gpuv100" + '\n')
				f.write("module load cuda10.2" + '\n')
				f.write("module load kaldi" + '\n')
				f.write("module load cudnn7.6-cuda10.2" + '\n')
				f.write('\n')
				f.write('cd /data/liuaal/asr_lm_size/' + '\n')
				f.write('\n')
				f.write('bash cmds/hupa_random_lexicon.sh ' + 'top ' + size + ' ' + str(n) + '\n')
				f.write('\n')


for lg in ['hupa']:
	for n in range(1, 6):
		for merge in ['200', '500', '1000', '1500', '2000', '2500', '3000', '3500', '4000', '4500', '5000']:
			for lm_order in ['5', '10', '15', '20']:
				for size in ['base', 'large']:
					for quality in ['top', 'second']:
						with open('bpe_pbs/' + lg + '_' + quality + '_bpe_' + merge + '_' + lm_order + '_' + size + '_' + str(n) + '.pbs', 'w') as f:
							f.write("#!/bin/bash" + '\n')
							f.write("#SBATCH --job-name='" + lg + " random'" + '\n')
							f.write("#SBATCH --ntasks 1 --cpus-per-task 4" + '\n')
							f.write("#SBATCH --mem=55gb" + '\n')
							f.write("#SBATCH --time=48:00:00" + '\n')
							f.write("#SBATCH --mail-type=BEGIN,END,FAIL." + '\n')
							f.write("#SBATCH --partition=gpuv100" + '\n')
							f.write("module load cuda10.2" + '\n')
							f.write("module load kaldi" + '\n')
							f.write("module load cudnn7.6-cuda10.2" + '\n')
							f.write('\n')
							f.write('cd /data/liuaal/asr_lm_size/' + '\n')
							f.write('\n')
							f.write('bash cmds/hupa_random_bpe.sh ' + merge + ' ' + quality + ' ' + size + ' ' + str(n) + ' ' + lm_order '\n')
							f.write('\n')

					for proportion in ['16']:
						size = 'proportion' + proportion + '.' + str(n)
						with open('bpe_pbs/' + lg + '_top_bpe_' + merge + '_' + lm_order + '_' + size + '.pbs', 'w') as f:
							f.write("#!/bin/bash" + '\n')
							f.write("#SBATCH --job-name='" + lg + " random'" + '\n')
							f.write("#SBATCH --ntasks 1 --cpus-per-task 4" + '\n')
							f.write("#SBATCH --mem=55gb" + '\n')
							f.write("#SBATCH --time=48:00:00" + '\n')
							f.write("#SBATCH --mail-type=BEGIN,END,FAIL." + '\n')
							f.write("#SBATCH --partition=gpuv100" + '\n')
							f.write("module load cuda10.2" + '\n')
							f.write("module load kaldi" + '\n')
							f.write("module load cudnn7.6-cuda10.2" + '\n')
							f.write('\n')
							f.write('cd /data/liuaal/asr_lm_size/' + '\n')
							f.write('\n')
							f.write('bash cmds/hupa_random_bpe_proportion.sh ' + merge + ' ' + str(n) + ' ' + proportion + ' ' + lm_order + '\n')
							f.write('\n')






