#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
from io import open
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import logging

import numpy as np
import theano
import six.moves.cPickle
import os, re, json

from keras.preprocessing import sequence, text
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils, generic_utils
from keras.models import Sequential
from keras.layers.embeddings import WordContextProduct, Embedding

from deepwalk import graph
from deepwalk import walks as serialized_walks
from gensim.models import Word2Vec
from skipgram import Skipgram

from six import text_type as unicode
from six import iteritems
from six.moves import range
from graph import build_deepwalk_corpus_minibatch_iter

import psutil
from multiprocessing import cpu_count

p = psutil.Process(os.getpid())
p.set_cpu_affinity(list(range(cpu_count())))

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def debug(type_, value, tb):
  if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    sys.__excepthook__(type_, value, tb)
  else:
    import traceback
    import pdb
    traceback.print_exception(type_, value, tb)
    print(u"\n")
    pdb.pm()

def process(args):

  print "Loading graph..."
  if args.format == "adjlist":
    G = graph.load_adjacencylist(args.input, undirected=args.undirected)
  elif args.format == "edgelist":
    G = graph.load_edgelist(args.input, undirected=args.undirected)
  elif args.format == "mat":
    G = graph.load_matfile(args.input, variable_name=args.matfile_variable_name, undirected=args.undirected)
  else:
    raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat'" % args.format)

  print("Number of nodes: {}".format(len(G.nodes())))

  num_walks = len(G.nodes()) * args.number_walks

  print("Number of walks: {}".format(num_walks))

  data_size = num_walks * args.walk_length

  print("Data size (walks*length): {}".format(data_size))

  if data_size < args.max_memory_data_size:
    #print("Walking...")
    #walks = graph.build_deepwalk_corpus(G, num_paths=args.number_walks,
    #                                    path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))
    print("Training...")
    max_features = len(G.nodes())  # vocabulary size
    dim_proj = args.representation_size  # embedding space dimension
    nb_epoch = 1   # number of training epochs

    # Neural network ( in Keras )
    model = Sequential()
    model.add(WordContextProduct(max_features, proj_dim=dim_proj, init="uniform"))
    model.compile(loss='mse', optimizer='rmsprop')
    sampling_table = sequence.make_sampling_table(max_features)

    print("Fitting tokenizer on walks...")
    tokenizer = text.Tokenizer(nb_words=max_features)

    print "Epochs: %d" % nb_epoch
    #tokenizer.fit_on_texts( build_deepwalk_corpus_minibatch_iter(G, args.number_walks, args.walk_length))

    for e in range(nb_epoch):
        print('-'*40)
        print('Epoch', e)
        print('-'*40)

        #progbar = generic_utils.Progbar(tokenizer.document_count)
        samples_seen = 0
        losses = []

#        for i, seq in enumerate(tokenizer.texts_to_sequences_generator( build_deepwalk_corpus_minibatch_iter(G, args.number_walks, args.walk_length) )):

        for i, seq in enumerate( build_deepwalk_corpus_minibatch_iter(G, args.number_walks, args.walk_length) ):
            # get skipgram couples for one text in the dataset
            couples, labels = sequence.skipgrams(seq, max_features, window_size=5, negative_samples=1., sampling_table=sampling_table)
            if couples:
                # one gradient update per sentence (one sentence = a few 1000s of word couples)
                X = np.array(couples, dtype="int32")
                print "Started fitting..."
                loss = model.fit(X, labels)

                print "Dumping..."

                # Dump weights to a temp file
                weights = model.layers[0].get_weights()[0]

                norm_weights = np_utils.normalize(weights)

                # TODO: save weights with indices
                np.savetxt( args.output, norm_weights )

                losses.append(loss)
                if len(losses) % 100 == 0:
    #                progbar.update(i, values=[("loss", np.mean(losses))])
                    losses = []
                samples_seen += len(labels)
        print('Samples seen:', samples_seen)
    print("Training completed!")

  else:
    print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, args.max_memory_data_size))
    print("Walking...")

    #TODO: IMPLEMENT THAT
    print "Not implemented yet..."
    sys.exit(1)

  print "Optimization done. Saving..."
  # recover the embedding weights trained with skipgram:
  weights = model.layers[0].get_weights()[0]

  # we no longer need this
  del model

  norm_weights = np_utils.normalize(weights)

  # TODO: save weights with indices
  np.savetxt( args.output, norm_weights )
  print "Saved!"

def main():
  parser = ArgumentParser("deepwalk",
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve')

  parser.add_argument("--debug", dest="debug", action='store_true', default=False,
                      help="drop a debugger if an exception is raised.")

  parser.add_argument('--format', default='adjlist',
                      help='File format of input file')

  parser.add_argument('--input', nargs='?', required=True,
                      help='Input graph file')

  parser.add_argument("-l", "--log", dest="log", default="INFO",
                      help="log verbosity level")

  parser.add_argument('--matfile-variable-name', default='network',
                      help='variable name of adjacency matrix inside a .mat file.')

  parser.add_argument('--max-memory-data-size', default=1000000000, type=int,
                      help='Size to start dumping walks to disk, instead of keeping them in memory.')

  parser.add_argument('--number-walks', default=10, type=int,
                      help='Number of random walks to start at each node')

  parser.add_argument('--output', required=True,
                      help='Output representation file')

  parser.add_argument('--representation-size', default=64, type=int,
                      help='Number of latent dimensions to learn for each node.')

  parser.add_argument('--seed', default=0, type=int,
                      help='Seed for random walk generator.')

  parser.add_argument('--undirected', default=True, type=bool,
                      help='Treat graph as undirected.')

  parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
                      help='Use vertex degree to estimate the frequency of nodes '
                           'in the random walks. This option is faster than '
                           'calculating the vocabulary.')

  parser.add_argument('--walk-length', default=40, type=int,
                      help='Length of the random walk started at each node')

  parser.add_argument('--window-size', default=5, type=int,
                      help='Window size of skipgram model.')

  parser.add_argument('--workers', default=1, type=int,
                      help='Number of parallel processes.')


  args = parser.parse_args()
  numeric_level = getattr(logging, args.log.upper(), None)
  logging.basicConfig(format=LOGFORMAT)
  logger.setLevel(numeric_level)

  if args.debug:
   sys.excepthook = debug

  process(args)

if __name__ == "__main__":
  sys.exit(main())
