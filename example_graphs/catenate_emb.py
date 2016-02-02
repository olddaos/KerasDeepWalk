#! /usr/bin/python

import sys
from sys import stdin

def main(argv):
	f1 = open(argv[0])
	f2 = open(argv[1])

	for line in f1:
		line2 = " ".join( f2.next().split()[1:] )
		print line.rstrip() + " " + line2

if __name__ == "__main__":
   main(sys.argv[1:])

