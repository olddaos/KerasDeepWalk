#! /usr/bin/python

import sys
from sys import stdin
import math

def main(argv):
	f1 = open(argv[0])

#	linecount = 0
	for line in f1:
		# Skip header
#		if linecount == 0:
#			print line.rstrip()
#			linecount = 1
#			continue

		vector_unnorm_str = line.rstrip().split()
		vertex      = vector_unnorm_str[0]
		vector_unnorm = map( lambda x: float(x), vector_unnorm_str[1:])
		vector_len = reduce( lambda x,y: x +y , map( lambda x: x*x, vector_unnorm[1:] ))

		vector_len  = math.sqrt( vector_len )
		try:
			vector_norm = [ vertex ]
			vector_norm.extend( [ str(x/vector_len) for x in vector_unnorm[1:] ] )

			print " ".join(  vector_norm )
		except:
			print " ".join( vector_unnorm_str )

if __name__ == "__main__":
   main(sys.argv[1:])

