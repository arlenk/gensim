#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""Corpus in the Matrix Market format.

This code uses python's struct library to read/write binary data

"""

import logging
import array

from gensim import utils


logger = logging.getLogger(__name__)


class MmReaderArray(object):
    """Matrix market file reader, used for :class:`~gensim.corpora.mmcorpus.MmCorpus`.

    Wrap a term-document matrix on disk (in matrix-market format), and present it
    as an object which supports iteration over the rows (~documents).

    Attributes
    ----------
    num_docs : int
        number of documents in market matrix file
    num_terms : int
        number of terms
    num_nnz : int
        number of non-zero terms

    Notes
    ----------
    Note that the file is read into memory one document at a time, not the whole matrix at once
    (unlike :meth:`~scipy.io.mmread`). This allows us to process corpora which are larger than the available RAM.

    """

    def __init__(self, input, transposed=True):
        """

        Parameters
        ----------
        input : {str, file-like object}
            Path to input file in MM format or a file-like object that supports `seek()`
            (e.g. :class:`~gzip.GzipFile`, :class:`~bz2.BZ2File`).

        transposed : bool, optional
            if True, expects lines to represent doc_id, term_id, value. Else, expects term_id, doc_id, value.

        """
        logger.info("initializing corpus reader from %s", input)
        self.input, self.transposed = input, transposed
        self.num_docs, self.num_terms, self.num_nnz = self.read_headers()

        logger.info(
            "accepted corpus with %i documents, %i features, %i non-zero entries",
            self.num_docs, self.num_terms, self.num_nnz
        )

    def __len__(self):
        """Get size of corpus (number of documents)."""
        return self.num_docs

    def __str__(self):
        return ("MmCorpus(%i documents, %i features, %i non-zero entries)" %
                (self.num_docs, self.num_terms, self.num_nnz))

    def skip_headers(self, input_file):
        """Skip file headers that appear before the first document.

        Parameters
        ----------
        input_file : iterable of str
            Iterable taken from file in MM format.

        """
        header = array.array('i')
        header.fromfile(input_file, 3)

    def read_headers(self):
        """Reader header row for file metadata

        Returns
        ----------
        num_docs : int
        num_terms : int
        num_nnz : int

        """

        with utils.smart_open(self.input, 'rb') as f:
            header = array.array('i')
            header.fromfile(f, 3)
            num_docs, num_terms, num_nnz = header

        return num_docs, num_terms, num_nnz

    @staticmethod
    def save_corpus(fname, corpus):
        logger.info("storing corpus in Matrix Market format to %s", fname)
        f = utils.smart_open(fname, 'wb')

        # write out header info
        num_docs, num_terms, num_nnz = corpus.num_docs, corpus.num_terms, corpus.num_nnz
        header = array.array("i", [ num_docs, num_terms, num_nnz ])
        header.tofile(f)

        for (docid, doc) in enumerate(corpus):
            doc_length = len(doc)
            doc_header = array.array("i", [ docid, doc_length])
            doc_header.tofile(f)

            # will store document as array of termids
            # followed by array of values
            termids = array.array('i')
            values = array.array('f')
            for (termid, value) in doc:
                termids.append(termid)
                values.append(value)

            termids.tofile(f)
            values.tofile(f)

        f.close()

    def __iter__(self):
        """Iterate through corpus.

        Notes
        ------
        Note that the total number of vectors returned is always equal to the number of rows specified
        in the header, empty documents are inserted and yielded where appropriate, even if they are not explicitly
        stored in the Matrix Market file.

        Yields
        ------
        (int, list of (int, number))
            Document id and Document in BoW format

        """

        with utils.smart_open(self.input, 'rb') as f:
            self.skip_headers(f)

            for _ in range(self.num_docs):
                doc_header = array.array('i')
                doc_header.fromfile(f, 2)
                docid, doc_length = doc_header

                termids = array.array('i')
                values = array.array('f')

                termids.fromfile(f, doc_length)
                values.fromfile(f, doc_length)
                document = zip(termids, values)

                yield document

