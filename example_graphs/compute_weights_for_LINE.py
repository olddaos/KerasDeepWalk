#! /usr/bin/python

import sys, getopt
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def main(argv):
   inputfile = ''
   outputfile = ''
   huge_true_set = set()
   huge_my_set  = set() 

   true_array = []
   my_array  = []
   try:
      opts, args = getopt.getopt(argv,"ho:e:",["ofile=","efile="])
   except getopt.GetoptError:
      print 'compare_communities.py -o <vertex_to_orbits> -e <edge_list>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'compare_communities.py -o <vertex_to_orbits> -e <edge_list>'
         sys.exit()
      elif opt in ("-o", "--ofile"):
         vertex2orbits = arg
      elif opt in ("-e", "--efile"):
         edgelist = arg
#   print 'Vertices to sample file is "', gtruth
 #  print 'My own file is "',  myfile

   # Read in vertices to sample from embedding 
   v2o = np.loadtxt( vertex2orbits )
   G = defaultdict(lambda: set())
   for orb in v2o:
      for el in orb[1:]:
         G[orb[0]].add( el )

   # Maps vertex to its orbits in O(n)
   G_dict = {int(k): [x for x in v] for k,v in G.iteritems()}

   skipheader = 1
   # Read in graph edge by edge, compute similarity, output triple for LINE 
   with open( edgelist, 'r') as inf:
      for line in inf:
         if skipheader :
            skipheader = 0
            continue
         vertices = line.split()
         if ( vertices.__len__() > 0):
            x1 = G_dict[ int(vertices[0])]
            x2 = G_dict[ int(vertices[1])]

	    try:
	       dist = int(cosine_similarity( x1, x2 )[0][0] * 100) * 1000
               if ( dist == 0 ): dist = 1

	    except:
	       dist = 1

	    print vertices[0] + " " + vertices[1] + " " + str(dist)
#	    print vertices[0] + " " + vertices[1] + " " + str(1)
if __name__ == "__main__":
   main(sys.argv[1:])
