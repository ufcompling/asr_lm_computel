import argparse, io, os

parser = argparse.ArgumentParser()
parser.add_argument('--lg', type = str, help = 'e.g., wolof')
parser.add_argument('--n', type = str, help = 'e.g., random split number')
parser.add_argument('--m', type = str, help = 'e.g., medium1, medium2')
args = parser.parse_args()

if args.lg != 'hupa':
	lm_file = io.open('cmds/lm_' + args.lg + '_random' + args.n + '.sh', 'w')
	top_tier_lm_file = io.open('cmds/lm_' + args.lg + '_top_tier_random' + args.n + '.sh', 'w')
	second_tier_lm_file = io.open('cmds/lm_' + args.lg + '_second_tier_random' + args.n + '.sh', 'w')
	top_tier_medium_lm_file = io.open('cmds/lm_' + args.lg + '_top_tier_random' + args.n + '_medium.sh', 'w')
	second_tier_medium_lm_file = io.open('cmds/lm_' + args.lg + '_second_tier_random' + args.n + '_medium.sh', 'w')

	base_file = io.open('cmds/' + args.lg + '_random' + args.n + '_base.sh', 'w')
	top_tier_base_file = io.open('cmds/' + args.lg + '_top_tier_random' + args.n + '_base.sh', 'w')
	second_tier_base_file = io.open('cmds/' + args.lg + '_second_tier_random' + args.n + '_base.sh', 'w')
	
	large_file = io.open('cmds/' + args.lg + '_random' + args.n + '_large.sh', 'w')
	top_tier_large_file = io.open('cmds/' + args.lg + '_top_tier_random' + args.n + '_large.sh', 'w')
	second_tier_large_file = io.open('cmds/' + args.lg + '_second_tier_random' + args.n + '_large.sh', 'w')
	
	top_tier_medium_file = io.open('cmds/' + args.lg + '_top_tier_random' + args.n + '_medium' + args.m + '.sh', 'w')
	second_tier_medium_file = io.open('cmds/' + args.lg + '_second_tier_random' + args.n + '_medium' + args.m + '.sh', 'w')

	base_dnn_file = io.open('cmds/' + args.lg + '_random' + args.n + '_base_dnn.sh', 'w')
	top_tier_base_dnn_file = io.open('cmds/' + args.lg + '_top_tier_random' + args.n + '_base_dnn.sh', 'w')
	second_tier_base_dnn_file = io.open('cmds/' + args.lg + '_second_tier_random' + args.n + '_base_dnn.sh', 'w')

	large_dnn_file = io.open('cmds/' + args.lg + '_random' + args.n + '_large_dnn.sh', 'w')
	top_tier_large_dnn_file = io.open('cmds/' + args.lg + '_top_tier_random' + args.n + '_large_dnn.sh', 'w')
	second_tier_large_dnn_file = io.open('cmds/' + args.lg + '_second_tier_random' + args.n + '_large_dnn.sh', 'w')
	
	top_tier_medium_dnn_file = io.open('cmds/' + args.lg + '_top_tier_random' + args.n + '_medium' + args.m + '_dnn.sh', 'w')
	second_tier_medium_dnn_file = io.open('cmds/' + args.lg + '_second_tier_random' + args.n + '_medium' + args.m + '_dnn.sh', 'w')


	with io.open('cmds/lm_fongbe_random1.sh') as f:
		for line in f:
			line = line.strip()
			line = line.replace('fongbe', args.lg)
			line = line.replace('train1', 'train' + args.n)
			line = line.replace('dev1', 'dev' + args.n)
			line = line.replace('system1', 'system' + args.n)
			lm_file.write(line + '\n')

	with io.open('cmds/lm_fongbe_top_tier_random1.sh') as f:
		for line in f:
			line = line.strip()
			line = line.replace('fongbe/top_tier/1/', 'fongbe/top_tier/' + args.n + '/')
			line = line.replace('fongbe', args.lg)
			top_tier_lm_file.write(line + '\n')

	with io.open('cmds/lm_fongbe_second_tier_random1.sh') as f:
		for line in f:
			line = line.strip()
			line = line.replace('fongbe/second_tier/1/', 'fongbe/second_tier/' + args.n + '/')
			line = line.replace('fongbe', args.lg)
			second_tier_lm_file.write(line + '\n')

	with io.open('cmds/lm_fongbe_top_tier_random1_medium.sh') as f:
		for line in f:
			line = line.strip()
			line = line.replace('fongbe', args.lg)
			line = line.replace('/1/random/', '/' + args.n + '/random/')
			top_tier_medium_lm_file.write(line + '\n')

	with io.open('cmds/lm_fongbe_second_tier_random1_medium.sh') as f:
		for line in f:
			line = line.strip()
			line = line.replace('fongbe', args.lg)
			line = line.replace('/1/random/', '/' + args.n + '/random/')
			second_tier_medium_lm_file.write(line + '\n')

	with io.open('cmds/fongbe_random1_base.sh') as f:
		for line in f:
			line = line.strip()
			line = line.replace('fongbe', args.lg)
			line = line.replace('train1', 'train' + args.n)
			line = line.replace('dev1', 'dev' + args.n)
			line = line.replace('system1', 'system' + args.n)
			line = line.replace('utt_spk1', 'utt_spk' + args.n)
			line = line.replace('compute_mfcc1', 'compute_mfcc' + args.n)
			base_file.write(line + '\n')

	with io.open('cmds/fongbe_top_tier_random1_base.sh') as f:
		for line in f:
			line = line.strip()
			line = line.replace('fongbe/top_tier/1/', 'fongbe/top_tier/' + args.n + '/')
			line = line.replace('fongbe', args.lg)
			top_tier_base_file.write(line + '\n')

	with io.open('cmds/fongbe_second_tier_random1_base.sh') as f:
		for line in f:
			line = line.strip()
			line = line.replace('fongbe/second_tier/1/', 'fongbe/second_tier/' + args.n + '/')
			line = line.replace('fongbe', args.lg)
			second_tier_base_file.write(line + '\n')

	with io.open('cmds/fongbe_random1_large.sh') as f:
		for line in f:
			line = line.strip()
			line = line.replace('fongbe', args.lg)
			line = line.replace('train1', 'train' + args.n)
			line = line.replace('dev1', 'dev' + args.n)
			line = line.replace('system1', 'system' + args.n)
			line = line.replace('utt_spk1', 'utt_spk' + args.n)
			line = line.replace('compute_mfcc1', 'compute_mfcc' + args.n)
			large_file.write(line + '\n')

	with io.open('cmds/fongbe_top_tier_random1_large.sh') as f:
		for line in f:
			line = line.strip()
			line = line.replace('fongbe/top_tier/1/', 'fongbe/top_tier/' + args.n + '/')
			line = line.replace('fongbe', args.lg)
			top_tier_large_file.write(line + '\n')

	with io.open('cmds/fongbe_second_tier_random1_large.sh') as f:
		for line in f:
			line = line.strip()
			line = line.replace('fongbe/second_tier/1/', 'fongbe/second_tier/' + args.n + '/')
			line = line.replace('fongbe', args.lg)
			second_tier_large_file.write(line + '\n')

	with io.open('cmds/fongbe_top_tier_random1_medium1.sh') as f:
		for line in f:
			line = line.strip()
			line = line.replace('fongbe/top_tier/1/', 'fongbe/top_tier/' + args.n + '/')
			line = line.replace('fongbe', args.lg)
			line = line.replace('medium1', 'medium' + args.m)
			top_tier_medium_file.write(line + '\n')

	with io.open('cmds/fongbe_second_tier_random1_medium1.sh') as f:
		for line in f:
			line = line.strip()
			line = line.replace('fongbe/second_tier/1/', 'fongbe/second_tier/' + args.n + '/')
			line = line.replace('fongbe', args.lg)
			line = line.replace('medium1', 'medium' + args.m)
			second_tier_medium_file.write(line + '\n')

	with io.open('cmds/fongbe_random1_base_dnn.sh') as f:
		for line in f:
			line = line.strip()
			line = line.replace('fongbe', args.lg)
			line = line.replace('train1', 'train' + args.n)
			line = line.replace('dev1', 'dev' + args.n)
			line = line.replace('system1', 'system' + args.n)
			line = line.replace('utt_spk1', 'utt_spk' + args.n)
			line = line.replace('compute_mfcc1', 'compute_mfcc' + args.n)
			base_dnn_file.write(line + '\n')

	with io.open('cmds/fongbe_top_tier_random1_base_dnn.sh') as f:
		for line in f:
			line = line.strip()
			line = line.replace('fongbe/top_tier/1/', 'fongbe/top_tier/' + args.n + '/')
			line = line.replace('fongbe', args.lg)
			top_tier_base_dnn_file.write(line + '\n')

	with io.open('cmds/fongbe_second_tier_random1_base_dnn.sh') as f:
		for line in f:
			line = line.strip()
			line = line.replace('fongbe/second_tier/1/', 'fongbe/second_tier/' + args.n + '/')
			line = line.replace('fongbe', args.lg)
			second_tier_base_dnn_file.write(line + '\n')

	with io.open('cmds/fongbe_random1_large_dnn.sh') as f:
		for line in f:
			line = line.strip()
			line = line.replace('fongbe', args.lg)
			line = line.replace('train1', 'train' + args.n)
			line = line.replace('dev1', 'dev' + args.n)
			line = line.replace('system1', 'system' + args.n)
			line = line.replace('utt_spk1', 'utt_spk' + args.n)
			line = line.replace('compute_mfcc1', 'compute_mfcc' + args.n)
			large_dnn_file.write(line + '\n')

	with io.open('cmds/fongbe_top_tier_random1_large_dnn.sh') as f:
		for line in f:
			line = line.strip()
			line = line.replace('fongbe/top_tier/1/', 'fongbe/top_tier/' + args.n + '/')
			line = line.replace('fongbe', args.lg)
			top_tier_large_dnn_file.write(line + '\n')

	with io.open('cmds/fongbe_second_tier_random1_large_dnn.sh') as f:
		for line in f:
			line = line.strip()
			line = line.replace('fongbe/second_tier/1/', 'fongbe/second_tier/' + args.n + '/')
			line = line.replace('fongbe', args.lg)
			second_tier_large_dnn_file.write(line + '\n')

	with io.open('cmds/fongbe_top_tier_random1_medium1_dnn.sh') as f:
		for line in f:
			line = line.strip()
			line = line.replace('fongbe/top_tier/1/', 'fongbe/top_tier/' + args.n + '/')
			line = line.replace('fongbe', args.lg)
			line = line.replace('medium1', 'medium' + args.m)
			top_tier_medium_file.write(line + '\n')

	with io.open('cmds/fongbe_second_tier_random1_medium1_dnn.sh') as f:
		for line in f:
			line = line.strip()
			line = line.replace('fongbe/second_tier/1/', 'fongbe/second_tier/' + args.n + '/')
			line = line.replace('fongbe', args.lg)
			line = line.replace('medium1', 'medium' + args.m)
			second_tier_medium_file.write(line + '\n')

else:
	for tier in ['top_tier', 'second_tier']:
		lm_file = io.open('cmds/lm_' + args.lg + '_' + tier + '_random' + args.n + '.sh', 'w')
		base_file = io.open('cmds/' + args.lg + '_' + tier + '_random' + args.n + '_base.sh', 'w')
		large_file = io.open('cmds/' + args.lg + '_' + tier + '_random' + args.n + '_large.sh', 'w')
		base_dnn_file = io.open('cmds/' + args.lg + '_' + tier + '_random' + args.n + '_base_dnn.sh', 'w')
		large_dnn_file = io.open('cmds/' + args.lg + '_' + tier + '_random' + args.n + '_large_dnn.sh', 'w')

		with io.open('cmds/lm_hupa_' + tier + '_random1.sh') as f:
			for line in f:
				line = line.strip()
				line = line.replace('train1', 'train' + args.n)
				line = line.replace('dev1', 'dev' + args.n)
				line = line.replace('system1', 'system' + args.n)
				lm_file.write(line + '\n')

		with io.open('cmds/hupa_' + tier + '_random1_base.sh') as f:
			for line in f:
				line = line.strip()
				line = line.replace('train1', 'train' + args.n)
				line = line.replace('dev1', 'dev' + args.n)
				line = line.replace('system1', 'system' + args.n)
				line = line.replace('utt_spk1', 'utt_spk' + args.n)
				line = line.replace('compute_mfcc1', 'compute_mfcc' + args.n)
				base_file.write(line + '\n')

		with io.open('cmds/hupa_' + tier + '_random1_large.sh') as f:
			for line in f:
				line = line.replace('train1', 'train' + args.n)
				line = line.replace('dev1', 'dev' + args.n)
				line = line.replace('system1', 'system' + args.n)
				line = line.replace('utt_spk1', 'utt_spk' + args.n)
				line = line.replace('compute_mfcc1', 'compute_mfcc' + args.n)
				large_file.write(line + '\n')


		with io.open('cmds/hupa_' + tier + '_random1_base_dnn.sh') as f:
			for line in f:
				line = line.strip()
				line = line.replace('train1', 'train' + args.n)
				line = line.replace('dev1', 'dev' + args.n)
				line = line.replace('system1', 'system' + args.n)
				line = line.replace('utt_spk1', 'utt_spk' + args.n)
				line = line.replace('compute_mfcc1', 'compute_mfcc' + args.n)
				base_dnn_file.write(line + '\n')

		with io.open('cmds/hupa_' + tier + '_random1_large_dnn.sh') as f:
			for line in f:
				line = line.strip()
				line = line.replace('train1', 'train' + args.n)
				line = line.replace('dev1', 'dev' + args.n)
				line = line.replace('system1', 'system' + args.n)
				line = line.replace('utt_spk1', 'utt_spk' + args.n)
				line = line.replace('compute_mfcc1', 'compute_mfcc' + args.n)
				large_dnn_file.write(line + '\n')
