from __future__ import print_function
import sys
sys.path.append(".")

import os
import urllib
import logging, gensim, bz2
import gensim.downloader as api
import gensim.corpora as gc
import gensim.corpora.cython_binary as cb
import argparse

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def download_uci_corpus(name, outdir):
    """
    Download corpus data from UCI repository

    """
    url_root = "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/"

    outfile = os.path.join(outdir, "vocab.{}.txt".format(name))
    print("downloading {} vocab file to: {}".format(name, outfile))
    urllib.request.urlretrieve(url_root + "vocab.{}.txt".format(name),
                               filename=outfile)

    outfile = os.path.join(outdir, "docword.{}.txt.gz".format(name))
    print("downloading {} bag of words to: {}".format(name, outfile))
    urllib.request.urlretrieve(url_root + "docword.{}.txt.gz".format(name),
                               filename=outfile)


def load_ucicorpus(name, corpus_dir):
    """
    Load (already downloaded) UCI corpus

    """
    corpus_file = os.path.join(corpus_dir, 'docword.{}.txt.gz'.format(name))
    vocab_file = os.path.join(corpus_dir, 'vocab.{}.txt'.format(name))

    if not os.path.exists(corpus_file):
        raise ValueError("corpus {} not found.. looking in: {}".format(name, corpus_file))

    if not os.path.exists(vocab_file):
        raise ValueError("vocab {} not found.. looking in: {}".format(name, vocab_file))

    corpus = gensim.corpora.UciCorpus(corpus_file, vocab_file)
    return corpus


def save_corpus(name, outdir):
    """
    Save corpus to outdir

    """
    if name == 'text8':
        texts = api.load(name)

        print("creating dictionary")
        dictionary = gc.Dictionary([[ word.lower() for word in text ] for text in texts ])
        outfile = os.path.join(outdir, "{}.dict".format(name))
        print("saving dictionary to: {}".format(outfile))
        dictionary.save(outfile)

        # standard mm corpus
        corpus = [ dictionary.doc2bow([ word.lower() for word in text]) for text in texts ]
    else:
        download_uci_corpus(name, outdir)
        corpus = load_ucicorpus(name, outdir)

    print("converting to mmcorpus")
    outfile = os.path.join(outdir, "{}.mm".format(name))
    print("saving corpus to {}".format(outfile))
    gc.MmCorpus.serialize(outfile, corpus)

    # binary mm corpus
    print("converting to binary mmcorpus")
    corpus = gc.MmCorpus(outfile)
    outfile = os.path.join(outdir, "{}.binary.mm".format(name))
    cb.MmReaderStructArray.save_corpus(outfile, corpus)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="download corpora for profiling")
    parser.add_argument("corpus", help="name of corpus to download", action="store")
    parser.add_argument("corpus_dir", help="destination directory to save corpus", action="store",)
    args = parser.parse_args()

    if args.corpus not in [ 'text8', 'nytimes' ]:
        raise ValueError("unknown corpus.. only text8 and nytimes supported")

    if not os.path.exists(args.corpus_dir):
        raise ValueError("corpus_dir ({}) does not exist".format(args.corpus_dir))

    print("downloading {} corpus to {}".format(args.corpus, args.corpus_dir))
    save_corpus(args.corpus, args.corpus_dir)


