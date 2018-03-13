from __future__ import print_function

import sys
sys.path.append('')

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from collections import OrderedDict
import os
import timeit
import argparse
import gensim.corpora.mmreader as mr_old
import gensim.matutils as mr_new
import gensim.corpora.cython_struct as sc
import gensim.corpora.python_struct as ps
import gensim.corpora.python_array as pa


def count_nnz(corpus):
    count = 0
    for doc in corpus:
        for term in doc:
            count += 1

    return count


def benchmark_corpus(corpus_class, corpus_file):
    """
    Benchmark corpus class using given file

    """

    if not os.path.exists(corpus_file):
        raise ValueError("could not find corpus file: {}".format(corpus_file))

    corpus = corpus_class(corpus_file)
    command = "count_nnz(corpus)"
    ns = dict(count_nnz=count_nnz, corpus=corpus)
    seconds = timeit.timeit(command, globals=ns, number=3)
    return seconds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="benchmark corpus")
    parser.add_argument("corpus", help="name of corpus to benchmark", action="store")
    parser.add_argument("corpus_dir", help="directory with corpus file", action="store",)
    args = parser.parse_args()

    if args.corpus not in [ 'text8', 'nytimes' ]:
        raise ValueError("unknown corpus.. only text8 and nytimes supported")

    corpus_file = os.path.join(args.corpus_dir, args.corpus)

    corpus_tests = [
        ("gensim 3.3 MmReader", mr_old.MmReader, corpus_file + ".mm"),
        ("gensim 3.4 MmReader", mr_new.MmReader, corpus_file + ".mm"),
        ("cython struct", sc.MmReaderStructArray, corpus_file + ".struct.mm"),
        ("cython struct (custom iterator)", sc.MmReaderStructArrayDocumentIterator, corpus_file + ".struct.mm"),
        ("cython struct (read only)", sc.MmReaderStructArrayReadOnly, corpus_file + ".struct.mm"),
        ("python struct", ps.MmReaderStructArray, corpus_file + ".struct.mm"),
        ("python array", pa.MmReaderArray, corpus_file + ".array.mm"),
    ]

    res = OrderedDict()
    for test in corpus_tests:
        name, cls, corpus_file = test
        print("testing {}".format(cls))
        seconds = benchmark_corpus(cls, corpus_file)
        print("  took: {:.1f} seconds".format(seconds))
        res[name] = seconds

    print("")
    print("benchmark results")
    print("-" * 80)
    print("")
    for name, seconds in res.items():
        print("{:<40}: {:.1f} seconds".format(name, seconds))
