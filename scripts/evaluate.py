import io, os, argparse, statistics
from jiwer import wer, cer

parser = argparse.ArgumentParser()
parser.add_argument('--input', type = str, help = 'ASR model output')

args = parser.parse_args()

outfile = open('evaluation.txt', 'w')
header = ['Language', 'Split', 'Quality', 'Size', 'WER', 'OOV', 'Reduction', 'N', 'Merge', 'LM_order', 'CER', 'CER Reduction']
outfile.write('\t'.join(w for w in header) + '\n')

quality_map = {'top_tier': 'verified', 'second_tier': 'coarse'}

basic_results = []
simulated_results = []
bpe_results = []

def oov(lexicon_file, gold_file):
	lexicon = []
	with open(lexicon_file) as f:
		for line in f:
			toks = line.strip().split()
			lexicon.append(toks[0])
	
	dev_words = [] 
	with open(gold_file) as f:
		for line in f:
			toks = line.strip().split()
			for w in toks[1 : ]:
				dev_words.append(w)

	oov_n = 0
	for w in dev_words:
		if w not in lexicon:
			oov_n += 1

	return round(oov_n * 100 / len(dev_words), 2)

for lang_dir in os.listdir(args.input):
	if lang_dir in ['fongbe', 'wolof', 'iban', 'swahili', 'bemba']:
		print(lang_dir)
		for evaluate_dir in ['random']:

			all_base_wers = []
			all_base_cers = []
			all_base_oovs = []

			all_proportion13_wers = []
			all_proportion13_cers = []
			all_proportion13_oovs = []

			all_large_wers = []
			all_large_cers = []
			all_large_oovs = []

			for output_dir in os.listdir(args.input + lang_dir + '/' + evaluate_dir + '/'):
				n = output_dir.split('_')[0][6 : ]
				gold_file = 'data_lexicon/' + lang_dir + '/random/dev' + n + '/text'
				gold = []
				gold_dict = {}
				with io.open(gold_file, encoding = 'utf-8') as f:
					for line in f:
						utterance = line.strip().split()
						toks = utterance[1 : ]
						gold.append(' '.join(w for w in toks))
						gold_dict[utterance[0]] = toks
				
				base_cer = 0
				base_pred_file = 'exp_lexicon/' + lang_dir + '/' + evaluate_dir + '/system' + str(n) + '_base/dnn4b_pretrain-dbn_dnn_smbr/decode_dev_it6/log/decode.1.log'
				if os.path.exists(base_pred_file):
					base_pred = []
					with open(base_pred_file, encoding = 'utf-8') as f:
						for line in f:
							try:
								utterance = line.strip().split()
								if utterance[0] in gold_dict:
									toks = utterance[1: ]
									base_pred.append(' '.join(w for w in toks))
							except:
								pass

					base_cer = cer(gold, base_pred)

				proportion13_cer = 0
				proportion13_pred_file = 'exp_lexicon/' + lang_dir + '/' + evaluate_dir + '/system' + str(n) + '_proportion13.' + str(n) + '/dnn4b_pretrain-dbn_dnn_smbr/decode_dev_it6/log/decode.1.log'
				if os.path.exists(proportion13_pred_file):
					proportion13_pred = []
					with open(proportion13_pred_file, encoding = 'utf-8') as f:
						for line in f:
							try:
								utterance = line.strip().split()
								if utterance[0] in gold_dict:
									toks = utterance[1: ]
									proportion13_pred.append(' '.join(w for w in toks))
							except:
								pass

					proportion13_cer = cer(gold, proportion13_pred)

				large_cer = 0
				large_pred_file = 'exp_lexicon/' + lang_dir + '/' + evaluate_dir + '/system' + str(n) + '_large/dnn4b_pretrain-dbn_dnn_smbr/decode_dev_it6/log/decode.1.log'
				if os.path.exists(large_pred_file):
					large_pred = []
					with open(large_pred_file, encoding = 'utf-8') as f:
						for line in f:
							try:
								utterance = line.strip().split()
								if utterance[0] in gold_dict:
									toks = utterance[1: ]
									large_pred.append(' '.join(w for w in toks))
							except:
								pass

					large_cer = cer(gold, large_pred)
			
				base_wer = 0
				base_oov = 0

				proportion13_wer = 0
				proportion13_oov = 0

				large_wer = 0
				large_oov = 0

				if 'RESULTS' in os.listdir(args.input + lang_dir + '/' + evaluate_dir + '/' + output_dir + '/'):
					
					with io.open(args.input + lang_dir + '/' + evaluate_dir + '/' + output_dir + '/RESULTS', encoding = 'utf-8') as f:
						c = 0
						for line in f:
						#	if 'tri3b_mmi_b0.1/decode_dev/' in line:
							if '/dnn4b_pretrain-dbn_dnn_smbr/decode_dev_it6' in line:
								toks = line.split()
								if 'base' in output_dir:
									base_wer = float(toks[1])								
									lexicon_file = 'data_lexicon/' + lang_dir + '/random/train' + n + '/local/dict_base/lexicon.txt'							
								#	base_oov = oov(lexicon_file, gold_file)
									
								if 'proportion13' in output_dir:
									proportion13_wer = float(toks[1])
									proportion_idx = output_dir.split('_')[-1]
									lexicon_file = 'data_lexicon/' + lang_dir + '/random/train' + n + '/local/dict_' + proportion_idx + '/lexicon.txt'
								#	proportion13_oov = oov(lexicon_file, gold_file)									

								if 'large' in output_dir:
									large_wer = float(toks[1])
									lexicon_file = 'data_lexicon/' + lang_dir + '/random/train' + n + '/local/dict_large/lexicon.txt'
								#	large_oov = oov(lexicon_file, gold_file)									

							else:
								c += 1

						if c != 13:
							print(lang_dir + '/' + evaluate_dir + '/' + output_dir + ' DNN results not complete')
							print('\n')

				else:
					print(lang_dir + '/' + evaluate_dir + '/' + output_dir + ' Results EMPTY')
				
				if base_wer != 0:
					all_base_wers.append(base_wer)
				if proportion13_wer != 0:
					all_proportion13_wers.append(proportion13_wer)
				if large_wer != 0:
					all_large_wers.append(large_wer)

				if base_oov != 0:
					all_base_oovs.append(base_oov)
				if proportion13_oov != 0:
					all_proportion13_oovs.append(proportion13_oov)
				if large_oov != 0:
					all_large_oovs.append(large_oov)

				if base_cer != 0:
					all_base_cers.append(base_cer)
				if proportion13_cer != 0:
					all_proportion13_cers.append(proportion13_cer)
				if large_cer != 0:
					all_large_cers.append(large_cer)

			ave_base_wer = '-'
			try:
				ave_base_wer = round(statistics.mean(all_base_wers), 2)
			except:
				ave_base_wer = '-'

			ave_proportion13_wer = '_'
			try:
				ave_proportion13_wer = round(statistics.mean(all_proportion13_wers), 2)
			except:
				ave_proportion13_wer = '-'

			ave_large_wer = '-'
			try:
				ave_large_wer = round(statistics.mean(all_large_wers), 2)
			except:
				ave_large_wer = '-'

			ave_base_cer = '-'
			try:
				ave_base_cer = round(statistics.mean(all_base_cers), 2)
			except:
				ave_base_cer = '-'

			ave_proportion13_cer = '_'
			try:
				ave_proportion13_cer = round(statistics.mean(all_proportion13_cers), 2)
			except:
				ave_proportion13_cer = '-'

			ave_large_cer = '-'
			try:
				ave_large_cer = round(statistics.mean(all_large_cers), 2)
			except:
				ave_large_cer = '-'

			proportion13_reduction = '-'
			try:
				proportion13_reduction = -1 * round((ave_proportion13_wer - ave_base_wer) * 100 / ave_base_wer, 2)
			except:
				proportion13_reduction = '-'

			large_reduction = '-'
			try:
				large_reduction = -1 * round((ave_large_wer - ave_base_wer) * 100 / ave_base_wer, 2)
			except:
				large_reduction = '-'

			proportion13_reduction_cer = '-'
			try:
				proportion13_reduction_cer = -1 * round((ave_proportion13_cer - ave_base_cer) * 100 / ave_base_cer, 2)
			except:
				proportion13_reduction_cer = '-'

			large_reduction_cer = '-'
			try:
				large_reduction_cer = -1 * round((ave_large_cer - ave_base_cer) * 100 / ave_base_cer, 2)
			except:
				large_reduction_cer = '-'

			ave_base_oov = '-'
			try:
				ave_base_oov = round(statistics.mean(all_base_oovs))
			except:
				ave_base_oov = '-'

			ave_proportion13_oov = '_'
			try:
				ave_proportion13_oov = round(statistics.mean(all_proportion13_oovs))
			except:
				ave_proportion13_oov = '-'

			ave_large_oov = '-'
			try:
				ave_large_oov = round(statistics.mean(all_large_oovs))
			except:
				ave_large_oov = '-'

			basic_results.append(lang_dir + ' & ' + str(ave_base_wer) + ' & ' + str(ave_base_oov) + ' & ' + ' - ' + ' & ' + str(ave_proportion13_wer) + ' & ' + str(ave_proportion13_oov) + ' & ' + str(proportion13_reduction) + ' & ' + str(ave_large_wer) + ' & ' + str(ave_large_oov) + ' & ' + str(large_reduction) + ' \\\\')

			if len(all_base_wers) != 0:
				info = [lang_dir, evaluate_dir, 'NONE', 'base', ave_base_wer, ave_base_oov, '-', len(all_base_wers), 'NONE', 'NONE', ave_base_cer]
				outfile.write('\t'.join(str(w) for w in info) + '\n')

			if len(all_proportion13_wers) != 0:
				info = [lang_dir, evaluate_dir, 'NONE', 'proportion13', ave_proportion13_wer, ave_proportion13_oov, proportion13_reduction, len(all_proportion13_wers), 'NONE', 'NONE', ave_proportion13_cer, proportion13_reduction_cer]
				outfile.write('\t'.join(str(w) for w in info) + '\n')

			if len(all_large_wers) != 0:
				info = [lang_dir, evaluate_dir, 'NONE', 'large', ave_large_wer, ave_large_oov, large_reduction, len(all_large_wers), 'NONE', 'NONE', ave_large_cer, large_reduction_cer]
				outfile.write('\t'.join(str(w) for w in info) + '\n')

				
		### Getting results from BPE

		for merge in ['200', '500', '1000', '1500', '2000', '2500', '3000', '3500', '4000', '4500', '']:
			for evaluate_dir in os.listdir(args.input + lang_dir + '/'):
				if evaluate_dir.endswith('bpe_' + merge):
					for lm_order in ['5', '10', '15', '20']:
						all_bpe_base_wers = []
						all_bpe_base_oovs = []

						all_bpe_large_wers = []
						all_bpe_large_oovs = []

						for output_dir in os.listdir(args.input + lang_dir + '/' + evaluate_dir + '/'):
							if output_dir.endswith('_' + lm_order):
								bpe_base_wer = 0
								bpe_base_oov = 0

								bpe_large_wer = 0
								bpe_large_oov = 0

								if 'RESULTS' in os.listdir(args.input + lang_dir + '/' + evaluate_dir + '/' + output_dir + '/'):
									with io.open(args.input + lang_dir + '/' + evaluate_dir + '/' + output_dir + '/RESULTS', encoding = 'utf-8') as f:
										c = 0
										for line in f:
										#	if 'tri3b_mmi_b0.1/decode_dev/' in line:
											if '/dnn4b_pretrain-dbn_dnn_smbr/decode_dev_it6' in line:
												toks = line.split()
												if 'base' in output_dir:
													bpe_base_wer = float(toks[1])
												if 'large' in output_dir:
													bpe_large_wer = float(toks[1])
											else:
												c += 1

										if c != 13:
											print(lang_dir + '/' + evaluate_dir + '/' + output_dir + ' DNN results not complete')
											print('\n')

								else:
									print(lang_dir + '/' + evaluate_dir + '/' + output_dir + ' Results EMPTY')
				
								if bpe_base_wer != 0:
									all_bpe_base_wers.append(bpe_base_wer)
								if bpe_large_wer != 0:
									all_bpe_large_wers.append(bpe_large_wer)

						if len(all_bpe_base_wers) != 0:
							info = [lang_dir, evaluate_dir, 'NONE', 'base', statistics.mean(all_bpe_base_wers), '-', len(all_bpe_base_wers), merge, lm_order]
							outfile.write('\t'.join(str(w) for w in info) + '\n')

						if len(all_bpe_large_wers) != 0:
							info = [lang_dir, evaluate_dir, 'NONE', 'large', statistics.mean(all_bpe_large_wers), '-', len(all_bpe_large_wers), merge, lm_order]
							outfile.write('\t'.join(str(w) for w in info) + '\n')

		for quality in ['top_tier', 'second_tier']:
			all_base_wers = []
			all_base_oovs = []
			all_base_cers = []

			all_proportion13_wers = []
			all_proportion13_oovs = []
			all_proportion13_cers = []

			all_large_wers = []
			all_large_oovs = []
			all_large_cers = []

			if os.path.exists(args.input + lang_dir + '/' + quality + '/'):
				for n in ['1', '2', '3']: #os.listdir(args.input + lang_dir + '/' + quality + '/'):
					if os.path.exists(args.input + lang_dir + '/' + quality + '/' + n + '/random/'):				
						for output_dir in os.listdir(args.input + lang_dir + '/' + quality + '/' + n + '/random/'):
							gold_file = 'data_lexicon/' + lang_dir + '/' + quality + '/' + n + '/random/dev' + str(n) + '/text'
							gold = []
							gold_dict = {}
							with io.open(gold_file, encoding = 'utf-8') as f:
								for line in f:
									utterance = line.strip().split()
									toks = utterance[1 : ]
									gold.append(' '.join(w for w in toks))
									gold_dict[utterance[0]] = toks

							base_cer = 0
							base_pred_file = 'exp_lexicon/' + lang_dir + '/' + quality + '/' + n + '/' + evaluate_dir + '/system' + str(n) + '_base/dnn4b_pretrain-dbn_dnn_smbr/decode_dev_it6/log/decode.1.log'
							if os.path.exists(base_pred_file):
								base_pred = []
								with open(base_pred_file, encoding = 'utf-8') as f:
									for line in f:
										try:
											utterance = line.strip().split()
											if utterance[0] in gold_dict:
												toks = utterance[1: ]
												base_pred.append(' '.join(w for w in toks))
										except:
											pass

								base_cer = cer(gold, base_pred)

							proportion13_cer = 0
							proportion13_pred_file = 'exp_lexicon/' + lang_dir + '/' + quality + '/' + n + '/' + evaluate_dir + '/system' + n + '_proportion13.' + n + '/dnn4b_pretrain-dbn_dnn_smbr/decode_dev_it6/log/decode.1.log'
							if os.path.exists(proportion13_pred_file):
								proportion13_pred = []
								with open(proportion13_pred_file, encoding = 'utf-8') as f:
									for line in f:
										try:
											utterance = line.strip().split()
											if utterance[0] in gold_dict:
												toks = utterance[1: ]
												proportion13_pred.append(' '.join(w for w in toks))
										except:
											pass

								proportion13_cer = cer(gold, proportion13_pred)

							large_cer = 0
							large_pred_file = 'exp_lexicon/' + lang_dir + '/' + quality + '/' + n + '/' + evaluate_dir + '/system' + str(n) + '_large/dnn4b_pretrain-dbn_dnn_smbr/decode_dev_it6/log/decode.1.log'
							if os.path.exists(large_pred_file):
								large_pred = []
								with open(large_pred_file, encoding = 'utf-8') as f:
									for line in f:
										try:
											utterance = line.strip().split()
											if utterance[0] in gold_dict:
												toks = utterance[1: ]
												large_pred.append(' '.join(w for w in toks))
										except:
											pass

								try:
									large_cer = cer(gold, base_pred)
								except:
									print(lang_dir, quality, n)

							base_wer = 0
							base_oov = 0

							proportion13_wer = 0
							proportion13_oov = 0

							large_wer = 0
							large_oov = 0

							if 'RESULTS' in os.listdir(args.input + lang_dir + '/' + quality + '/' + n + '/random/' + output_dir + '/'):						
								with io.open(args.input + lang_dir + '/' + quality + '/' + n + '/random/' + output_dir + '/RESULTS', encoding = 'utf-8') as f:
									c = 0
									for line in f:
								#	if 'tri3b_mmi_b0.1/decode_dev/' in line:
										if '/dnn4b_pretrain-dbn_dnn_smbr/decode_dev_it6' in line:
											toks = line.split()
											if 'base' in output_dir:
												base_wer = float(toks[1])								
												lexicon_file = 'data_lexicon/' + lang_dir + '/' + quality + '/' + n + '/random/train' + str(n) + '/local/dict_base/lexicon.txt'							
												base_oov = oov(lexicon_file, gold_file)
											

											if 'proportion13' in output_dir:
												proportion13_wer = float(toks[1])
												proportion_idx = output_dir.split('_')[-1]
												lexicon_file = 'data_lexicon/' + lang_dir + '/' + quality + '/' + n + '/random/train' + str(n) + '/local/dict_' + proportion_idx + '/lexicon.txt'
												proportion13_oov = oov(lexicon_file, gold_file)
											

											if 'large' in output_dir:
												large_wer = float(toks[1])
												lexicon_file = 'data_lexicon/' + lang_dir + '/' + quality + '/' + n + '/random/train' + str(n) + '/local/dict_large/lexicon.txt'
												large_oov = oov(lexicon_file, gold_file)
											

										else:
											c += 1

									if c != 13:
										print(lang_dir + '/' + quality + '/' + n + '/random/' + output_dir + ' DNN results not complete')
										print('\n')

							else:
								print(lang_dir + '/' + quality + '/' + n + '/random/' + output_dir + ' Results EMPTY')
				
							if base_wer != 0:
								all_base_wers.append(base_wer)
							if proportion13_wer != 0:
								all_proportion13_wers.append(proportion13_wer)
							if large_wer != 0:
								all_large_wers.append(large_wer)

							if base_oov != 0:
								all_base_oovs.append(base_oov)
							if proportion13_oov != 0:
								all_proportion13_oovs.append(proportion13_oov)
							if large_oov != 0:
								all_large_oovs.append(large_oov)

							if base_cer != 0:
								all_base_cers.append(base_cer)
							if proportion13_cer != 0:
								all_proportion13_cers.append(proportion13_cer)
							if large_cer != 0:
								all_large_cers.append(large_cer)

			ave_base_wer = '-'
			try:
				ave_base_wer = round(statistics.mean(all_base_wers), 2)
			except:
				ave_base_wer = '-'

			ave_proportion13_wer = '_'
			try:
				ave_proportion13_wer = round(statistics.mean(all_proportion13_wers), 2)
			except:
				ave_proportion13_wer = '-'

			ave_large_wer = '-'
			try:
				ave_large_wer = round(statistics.mean(all_large_wers), 2)
			except:
				ave_large_wer = '-'

			ave_base_cer = '-'
			try:
				ave_base_cer = round(statistics.mean(all_base_cers), 2)
			except:
				ave_base_cer = '-'

			ave_proportion13_cer = '_'
			try:
				ave_proportion13_cer = round(statistics.mean(all_proportion13_cers), 2)
			except:
				ave_proportion13_cer = '-'

			ave_large_cer = '-'
			try:
				ave_large_cer = round(statistics.mean(all_large_cers), 2)
			except:
				ave_large_cer = '-'

			proportion13_reduction = '-'
			try:
				proportion13_reduction = -1 * round((ave_proportion13_wer - ave_base_wer) * 100 / ave_base_wer, 2)
			except:
				proportion13_reduction = '-'

			large_reduction = '-'
			try:
				large_reduction = -1 * round((ave_large_wer - ave_base_wer) * 100 / ave_base_wer, 2)
			except:
				large_reduction = '-'

			proportion13_reduction_cer = '-'
			try:
				proportion13_reduction_cer = -1 * round((ave_proportion13_cer - ave_base_cer) * 100 / ave_base_cer, 2)
			except:
				proportion13_reduction_cer = '-'

			large_reduction_cer = '-'
			try:
				large_reduction_cer = -1 * round((ave_large_cer - ave_base_cer) * 100 / ave_base_cer, 2)
			except:
				large_reduction_cer = '-'

			ave_base_oov = '-'
			try:
				ave_base_oov = round(statistics.mean(all_base_oovs), 2)
			except:
				ave_base_oov = '-'

			ave_proportion13_oov = '_'
			try:
				ave_proportion13_oov = round(statistics.mean(all_proportion13_oovs), 2)
			except:
				ave_proportion13_oov = '-'

			ave_large_oov = '-'
			try:
				ave_large_oov = round(statistics.mean(all_large_oovs), 2)
			except:
				ave_large_oov = '-'

			simulated_results.append(lang_dir + ' & ' + quality_map[quality] + ' & ' + str(ave_base_wer) + ' & ' + str(ave_base_oov) + ' & ' + ' - ' + ' & ' + str(ave_proportion13_wer) + ' & ' + str(ave_proportion13_oov) + ' & ' + str(proportion13_reduction) + ' & ' + str(ave_large_wer) + ' & ' + str(ave_large_oov) + ' & ' + str(large_reduction) + ' \\\\')

			if len(all_base_wers) != 0:
				info = [lang_dir, 'random', quality, 'base', ave_base_wer, ave_base_oov, '-', len(all_base_wers), 'NONE', 'NONE', ave_base_cer]
				outfile.write('\t'.join(str(w) for w in info) + '\n')

			if len(all_proportion13_wers) != 0:
				info = [lang_dir, 'random', quality, 'proportion13', ave_proportion13_wer, ave_proportion13_oov, proportion13_reduction, len(all_proportion13_wers), 'NONE', 'NONE', ave_proportion13_cer, proportion13_reduction_cer]
				outfile.write('\t'.join(str(w) for w in info) + '\n')

			if len(all_large_wers) != 0:
				info = [lang_dir, 'random', quality, 'large', ave_large_wer, ave_large_oov, large_reduction, len(all_large_wers), 'NONE', 'NONE', ave_large_cer, large_reduction_cer]
				outfile.write('\t'.join(str(w) for w in info) + '\n')


	if lang_dir in ['hupa']:
		for quality in ['top_tier', 'second_tier']:
			for evaluate_dir in ['random']:

				all_base_wers = []
				all_base_oovs = []
				all_base_cers = []

				all_proportion13_wers = []
				all_proportion13_oovs = []
				all_proportion13_cers = []

				all_large_wers = []
				all_large_oovs = []
				all_large_cers = []

				for output_dir in os.listdir(args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/'):
					n = output_dir.split('_')[0][6 : ]					
					gold_file = 'data_lexicon/' + lang_dir + '/' + quality + '/random/dev' + n + '/text'
					gold = []
					gold_dict = {}
					with io.open(gold_file, encoding = 'utf-8') as f:
						for line in f:
							utterance = line.strip().split()
							toks = utterance[1 : ]
							gold.append(' '.join(w for w in toks))
							gold_dict[utterance[0]] = toks

					base_cer = 0
					base_pred_file = 'exp_lexicon/' + lang_dir + '/' + quality + '/' + evaluate_dir + '/system' + str(n) + '_base/dnn4b_pretrain-dbn_dnn_smbr/decode_dev_it6/log/decode.1.log'
					if os.path.exists(base_pred_file):
						base_pred = []
						with open(base_pred_file, encoding = 'utf-8') as f:
							for line in f:
								try:
									utterance = line.strip().split()
									if utterance[0] in gold_dict:
										toks = utterance[1: ]
										base_pred.append(' '.join(w for w in toks))
								except:
									pass

						base_cer = cer(gold, base_pred)

					proportion13_cer = 0
					proportion13_pred_file = 'exp_lexicon/' + lang_dir + '/' + quality + '/' + evaluate_dir + '/system' + n + '_proportion13.' + n + '/dnn4b_pretrain-dbn_dnn_smbr/decode_dev_it6/log/decode.1.log'
					if os.path.exists(proportion13_pred_file):
						proportion13_pred = []
						with open(proportion13_pred_file, encoding = 'utf-8') as f:
							for line in f:
								try:
									utterance = line.strip().split()
									if utterance[0] in gold_dict:
										toks = utterance[1: ]
										proportion13_pred.append(' '.join(w for w in toks))
								except:
									pass

						proportion13_cer = cer(gold, proportion13_pred)


					large_cer = 0
					large_pred_file = 'exp_lexicon/' + lang_dir + '/' + quality + '/' + evaluate_dir + '/system' + str(n) + '_large/dnn4b_pretrain-dbn_dnn_smbr/decode_dev_it6/log/decode.1.log'
					if os.path.exists(large_pred_file):
						large_pred = []
						with open(large_pred_file, encoding = 'utf-8') as f:
							for line in f:
								try:
									utterance = line.strip().split()
									if utterance[0] in gold_dict:
										toks = utterance[1: ]
										large_pred.append(' '.join(w for w in toks))
								except:
									pass

						large_cer = cer(gold, large_pred)


					base_wer = 0
					base_oov = 0

					proportion13_wer = 0
					proportion13_oov = 0

					large_wer = 0
					large_oov = 0

					if 'RESULTS' in os.listdir(args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + '/'):					
						with io.open(args.input + lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + '/RESULTS', encoding = 'utf-8') as f:
							c = 0
							for line in f:
								if '/dnn4b_pretrain-dbn_dnn_smbr/decode_dev_it6' in line:
									toks = line.split()
									if 'base' in output_dir:
										base_wer = float(toks[1])								
										lexicon_file = 'data_lexicon/' + lang_dir + '/' + quality + '/random/train' + n + '/local/dict_base/lexicon.txt'							
										base_oov = oov(lexicon_file, gold_file)
										
									if 'proportion13' in output_dir:
										proportion13_wer = float(toks[1])
										proportion_idx = output_dir.split('_')[-1]
										lexicon_file = 'data_lexicon/' + lang_dir + '/' + quality + '/random/train' + n + '/local/dict_' + proportion_idx + '/lexicon.txt'
										proportion13_oov = oov(lexicon_file, gold_file)
										
									if 'large' in output_dir:
										large_wer = float(toks[1])
										lexicon_file = 'data_lexicon/' + lang_dir + '/' + quality + '/random/train' + n + '/local/dict_large/lexicon.txt'
										large_oov = oov(lexicon_file, gold_file)
										

								else:
									c += 1

							if c != 13:
								print(lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + ' DNN results not complete')
								print('\n')

					else:
						print(lang_dir + '/' + quality + '/' + evaluate_dir + '/' + output_dir + ' Results EMPTY')
				
					if base_wer != 0:
						all_base_wers.append(base_wer)
					if proportion13_wer != 0:
						all_proportion13_wers.append(proportion13_wer)
					if large_wer != 0:
						all_large_wers.append(large_wer)

					if base_oov != 0:
						all_base_oovs.append(base_oov)
					if proportion13_oov != 0:
						all_proportion13_oovs.append(proportion13_oov)
					if large_oov != 0:
						all_large_oovs.append(large_oov)

					if base_cer != 0:
						all_base_cers.append(base_cer)
					if proportion13_wer != 0:
						all_proportion13_cers.append(proportion13_cer)
					if large_cer != 0:
						all_large_cers.append(large_cer)
				
				ave_base_wer = '-'
				try:
					ave_base_wer = round(statistics.mean(all_base_wers), 2)
				except:
					ave_base_wer = '-'

				ave_proportion13_wer = '_'
				try:
					ave_proportion13_wer = round(statistics.mean(all_proportion13_wers), 2)
				except:
					ave_proportion13_wer = '-'

				ave_large_wer = '-'
				try:
					ave_large_wer = round(statistics.mean(all_large_wers), 2)
				except:
					ave_large_wer = '-'

				ave_base_cer = '-'
				try:
					ave_base_cer = round(statistics.mean(all_base_cers), 2)
				except:
					ave_base_cer = '-'

				ave_proportion13_cer = '_'
				try:
					ave_proportion13_cer = round(statistics.mean(all_proportion13_cers), 2)
				except:
					ave_proportion13_cer = '-'

				ave_large_cer = '-'
				try:
					ave_large_cer = round(statistics.mean(all_large_cers), 2)
				except:
					ave_large_cer = '-'

				proportion13_reduction = '-'
				try:
					proportion13_reduction = -1 * round((ave_proportion13_wer - ave_base_wer) * 100 / ave_base_wer, 2)
				except:
					proportion13_reduction = '-'

				large_reduction = '-'
				try:
					large_reduction = -1 * round((ave_large_wer - ave_base_wer) * 100 / ave_base_wer, 2)
				except:
					large_reduction = '-'

				proportion13_reduction_cer = '-'
				try:
					proportion13_reduction_cer = -1 * round((ave_proportion13_cer - ave_base_cer) * 100 / ave_base_cer, 2)
				except:
					proportion13_reduction_cer = '-'

				large_reduction_cer = '-'
				try:
					large_reduction_cer = -1 * round((ave_large_cer - ave_base_cer) * 100 / ave_base_cer, 2)
				except:
					large_reduction_cer = '-'

				ave_base_oov = '-'
				try:
					ave_base_oov = round(statistics.mean(all_base_oovs), 2)
				except:
					ave_base_oov = '-'

				ave_proportion13_oov = '_'
				try:
					ave_proportion13_oov = round(statistics.mean(all_proportion13_oovs), 2)
				except:
					ave_proportion13_oov = '-'

				ave_large_oov = '-'
				try:
					ave_large_oov = round(statistics.mean(all_large_oovs), 2)
				except:
					ave_large_oov = '-'

				basic_results.append(lang_dir + ' (' + quality_map[quality] + ')' + ' & ' + str(ave_base_wer) + ' & ' + str(ave_base_oov) + ' & ' + ' - ' + ' & ' + str(ave_proportion13_wer) + ' & ' + str(ave_proportion13_oov) + ' & ' + str(proportion13_reduction) + ' & ' + str(ave_large_wer) + ' & ' + str(ave_large_oov) + ' & ' + str(large_reduction) + ' \\\\')

				if len(all_base_wers) != 0:
					info = [lang_dir, evaluate_dir, quality, 'base', ave_base_wer, ave_base_oov, '-', len(all_base_wers), 'NONE', 'NONE', ave_base_cer]
					outfile.write('\t'.join(str(w) for w in info) + '\n')

				if len(all_proportion13_wers) != 0:
					info = [lang_dir, evaluate_dir, quality, 'proportion13', ave_proportion13_wer, ave_proportion13_oov, proportion13_reduction, len(all_proportion13_wers), 'NONE', 'NONE', ave_proportion13_cer, proportion13_reduction_cer]
					outfile.write('\t'.join(str(w) for w in info) + '\n')

				if len(all_large_wers) != 0:
					info = [lang_dir, evaluate_dir, quality, 'large', ave_large_wer, ave_large_oov, large_reduction, len(all_large_wers), 'NONE', 'NONE', ave_large_cer, large_reduction_cer]
					outfile.write('\t'.join(str(w) for w in info) + '\n')


print('\n')
print('\n')
print('\n')
print('\n')

### Generating tables ###

print("\\begin{table*}[ht!]")
print("\\centering")
print("\\footnotesize")
print("\\begin{tabular}{llllllllll}")
print("\\multicolumn{1}{c}{\\textbf{Language}} & ")
print("\\multicolumn{3}{c}{\\textbf{LM\_base}} & ")
print("\\multicolumn{3}{c}{\\textbf{LM\_prop}} &")
print("\\multicolumn{3}{c}{\\textbf{LM\_large}} \\\\")
print("\\cmidrule(lr){2-4}")
print("\\cmidrule(lr){5-7}")
print("\\cmidrule(lr){8-10}  \\\\")
print("{} & WER (\%) & OOV & reduction (\%) & WER (\%) & OOV & reduction (\%) & WER (\%) & OOV & reduction (\%)   \\\\")
print("\\midrule")
for tok in basic_results:
	print(tok)
print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{WER results from different LM configurations.}")
print("\\label{basic}")
print("\\end{table*}")

print('\n')
print('\n')
print('\n')
print('\n')

print("\\begin{table*}[ht!]")
print("\\centering")
print("\\footnotesize")
print("\\begin{tabular}{lllllllllll}")
print("\\multicolumn{1}{c}{\\textbf{Language}} & ")
print("\\multicolumn{1}{c}{\\textbf{Setting}} &")
print("\\multicolumn{3}{c}{\\textbf{LM\_base}} & ")
print("\\multicolumn{3}{c}{\\textbf{LM\_prop}} &")
print("\\multicolumn{3}{c}{\\textbf{LM\_large}} \\\\")
print("\\cmidrule(lr){3-5}")
print("\\cmidrule(lr){6-8}")
print("\\cmidrule(lr){9-11}  \\")
print("{} & {} & WER (\%) & OOV & reduction (\%) &  WER (\%) & OOV & reduction (\%) &  WER (\%) & OOV & reduction (\%)  \\\\")
print("\\midrule")
for tok in simulated_results:
	print(tok)
print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{WER results from different LM configurations in simulated settings; verified and coarse refer to the simulated settings following the setup of Language H, and do not refer to the quality of the data.}")
print("\\label{simulated}")
print("\\end{table*}")











