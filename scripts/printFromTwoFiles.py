#!/usr/local/bin/python3.6

import sys, argparse, json, os
import os.path
from os import path

def main(argv):
    parser = argparse.ArgumentParser(description='Grab lines for images')
    parser.add_argument('-fileOne','--fileOne', help='file', required=True)
    parser.add_argument('-fileTwo','--fileTwo', help='file', required=True)
    args = parser.parse_args()
    argsdict = vars(args)
    fileOne = argsdict['fileOne']
    fileTwo = argsdict['fileTwo']

    trainFile = [line.rstrip('\n') for line in open(fileOne, mode='r', encoding='utf-8-sig')]
    trainFix = [line.rstrip('\n') for line in open(fileTwo, mode='r', encoding='utf-8-sig')]

    res = "\n".join("{} {}".format(x, y) for x, y in zip(trainFile, trainFix))	

    #print(res)
    for idx, line in enumerate(trainFile):
        #arrFile = trainFile[idx].split()
        arrFile = line.split(" ")
        print(arrFile[0] + " " + trainFix[idx])
        #print(arrFile[0])	

	
if __name__ == "__main__":
    main(sys.argv[1:])
