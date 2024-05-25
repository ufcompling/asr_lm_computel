#!/usr/local/bin/python3.6

import sys, argparse, json, os, re
import os.path
from os import path

def orthoToArp(word):
	Sen_arp = word

	#Sen_arp = Sen_arp.replace("'", "’")
	Sen_arp = re.sub(r'@', r'', Sen_arp)
	Sen_arp = re.sub(r'▁', r'', Sen_arp)
	Sen_arp = re.sub('_', '', Sen_arp)
	Sen_arp = re.sub('-', '', Sen_arp)
	Sen_arp = re.sub(r"'", r"’", Sen_arp)
	Sen_arp = re.sub(r'1', r'AA GS', Sen_arp)
	Sen_arp = re.sub(r'2', r'AE GS', Sen_arp)
	Sen_arp = re.sub(r'3', r'EY GS', Sen_arp)
	Sen_arp = re.sub(r'4', r'EH GS', Sen_arp)
	Sen_arp = re.sub(r'5', r'IY GS', Sen_arp)
	Sen_arp = re.sub(r'6', r'OW GS', Sen_arp)
	Sen_arp = re.sub(r'7', r'AO GS', Sen_arp)
	Sen_arp = re.sub(r'8', r'UW GS', Sen_arp)
	Sen_arp = re.sub(r'9', r'gs', Sen_arp)
	Sen_arp = re.sub(r":", r"", Sen_arp)
	Sen_arp = re.sub(r"kw", r"K w", Sen_arp)
	Sen_arp = re.sub(r"kn", r"K n", Sen_arp)
	Sen_arp = re.sub(r"kd", r"K d", Sen_arp)
	Sen_arp = re.sub(r"r", r"R ", Sen_arp)
	Sen_arp = re.sub(r"c", r"C ", Sen_arp)
	Sen_arp = re.sub(r"m", r"M ", Sen_arp)
	Sen_arp = re.sub(r"p", r"P ", Sen_arp)
	Sen_arp = re.sub(r"b", r"B ", Sen_arp)
	Sen_arp = re.sub(r"f", r"F ", Sen_arp)
	Sen_arp = re.sub(r"q", r"Q ", Sen_arp)
	Sen_arp = re.sub(r"ky", r"K y", Sen_arp)
	Sen_arp = re.sub(r"o", r"OW ", Sen_arp)
	#Sen_arp = re.sub('ö', 'AO ', Sen_arp)
	Sen_arp = re.sub(r"e", r"EY ", Sen_arp)
	#Sen_arp = re.sub(r"ë", r"EH ", Sen_arp)
	Sen_arp = re.sub(r"i", r"iy ", Sen_arp)
	Sen_arp = re.sub(r"u", r"UW ", Sen_arp)
	#Sen_arp = re.sub(r"ä", r"AE ", Sen_arp)
	Sen_arp = re.sub(r"a", r"AA ", Sen_arp)
	Sen_arp = re.sub(r"tš", r"ch ", Sen_arp)
	Sen_arp = re.sub(r"u", r"UW ", Sen_arp)
	Sen_arp = re.sub(r"w", r"w ", Sen_arp)
	Sen_arp = re.sub(r'ö', r'AO ', Sen_arp)
	Sen_arp = re.sub(r"ä", r"AE ", Sen_arp)
	Sen_arp = re.sub(r"ë", r"EH ", Sen_arp)

	Sen_arp = re.sub(r"j", r"JH ", Sen_arp)
	Sen_arp = re.sub(r"(?<!’)t$", r"d ", Sen_arp)
	Sen_arp = re.sub(r"t", r"t ", Sen_arp)
	Sen_arp = re.sub(r"d", r"d ", Sen_arp)
	Sen_arp = re.sub(r"g", r"g ", Sen_arp)
	#Sen_arp = re.sub(r"(?<!’)k$", r"g", Sen_arp)
	Sen_arp = re.sub(r"ö", r"AO ", Sen_arp)
	Sen_arp = re.sub(r"k", r"g ", Sen_arp)
	Sen_arp = re.sub(r"g $", r"k", Sen_arp)
	Sen_arp = re.sub(r"d $", r"t", Sen_arp)
	Sen_arp = re.sub(r"ts", r"T S ", Sen_arp)
	Sen_arp = re.sub(r"s", r"S ", Sen_arp)
	Sen_arp = re.sub(r"dz", r"D Z ", Sen_arp)
	Sen_arp = re.sub(r"n", r"n ", Sen_arp)
	Sen_arp = re.sub(r"y", r"y ", Sen_arp)
	Sen_arp = re.sub(r"h", r"hh ", Sen_arp)
	Sen_arp = re.sub(r"š", r"sh ", Sen_arp)
	Sen_arp = re.sub(r"z", r"z ", Sen_arp)
	Sen_arp = re.sub(r"’", r"GS ", Sen_arp)
	Sen_arp = re.sub(r"öEH", r"AO EH ", Sen_arp)
	Sen_arp = re.sub(r"än", r"AE N ", Sen_arp)
	Sen_arp = re.sub(r"g$", r"k", Sen_arp)

	Sen_arp = re.sub(r"ë", r"EH ", Sen_arp)
	Sen_arp = re.sub(r"äd", r"ae d", Sen_arp)

	Sen_arp = re.sub(r"chh", r"ch", Sen_arp)
	Sen_arp = re.sub(r"  ", r" ", Sen_arp)
	Sen_arp = re.sub(r"\+", r"", Sen_arp)

	return Sen_arp.rstrip()

def main(argv):
	parser = argparse.ArgumentParser(description='Grab lines for images')
	parser.add_argument('-wordList','--wordList', help='file', required=True)
	args = parser.parse_args()
	argsdict = vars(args)
	csvFile = argsdict['wordList']
	with open(csvFile) as fp:
		for line in fp:
			arpSeneca = orthoToArp(line)
			print(line.rstrip() + " " + arpSeneca.lower())

if __name__ == "__main__":
	main(sys.argv[1:])
