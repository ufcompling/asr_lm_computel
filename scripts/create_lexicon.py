#!/usr/local/bin/python3.6

import sys, argparse, json, os, re
import os.path
from os import path

def orthoToArp(line):
	word = line.strip()
	word = re.sub(r'@', r'', word)
	word = list(word)
	word = ' '.join(c.lower() for c in word)
	return word

def main(argv):
	parser = argparse.ArgumentParser(description='Grab lines for images')
	parser.add_argument('-wordList','--wordList', help='file', required=True)
	args = parser.parse_args()
	argsdict = vars(args)
	csvFile = argsdict['wordList']
	print("!SIL sil")
	print("<UNK> spn")
	with open(csvFile, encoding='latin-1') as fp:
		for line in fp:
			if line.strip() != '':
				sounds = orthoToArp(line)
				info = line.rstrip() + " " + sounds
				word_len = len(line.rstrip().split())
				digit_c = 0
				for n in info.split():
					if n.isdigit() is True:
						digit_c += 1
				if digit_c == 0 and word_len == 1 and len(info.split()) > 1:
					print(line.rstrip() + " " + sounds)

if __name__ == "__main__":
	main(sys.argv[1:])
