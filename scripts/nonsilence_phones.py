import sys

sounds = []
with open(sys.argv[1]) as f:
	for line in f:
		toks = line.strip().split()[1 : ]
		for s in toks:
			if s not in sounds:
				sounds.append(s)

for s in sounds:
	print(s)