import io, os

for lg in ['fongbe', 'wolof', 'iban', 'swahili']:
	outfile = open(lg + '_prep.sh', 'w')
	for n in range(1, 2):
		for merge in ['200', '500', '1000', '1500', '2000', '2500', '3000', '3500', '4000', '4500', '5000']:
			cmd = 'bash utt2spk_wav.sh ' + merge + ' ' + lg + ' ' + str(n)
			outfile.write(cmd + '\n')

			for lm_order in ['5', '10', '15', '20']:			
				cmd = 'bash bpe_base_prep.sh ' + merge + ' ' + lg + ' ' + str(n) + ' ' + lm_order
				outfile.write(cmd + '\n')

				cmd = 'bash bpe_large_prep.sh ' + merge + ' ' + lg + ' ' + str(n) + ' ' + lm_order
				outfile.write(cmd + '\n')
