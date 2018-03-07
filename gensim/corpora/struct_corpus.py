#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""Corpus in the Matrix Market format."""

import logging
import struct

from gensim import utils


logger = logging.getLogger(__name__)


class MmReader1(object):
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
        s = struct.Struct('i i i')
        num_docs, num_terms, num_nnz = s.unpack(input_file.read(s.size))

    def read_headers(self):
        """Reader header row for file metadata

        Returns
        ----------
        num_docs : int
        num_terms : int
        num_nnz : int

        """

        with utils.smart_open(self.input, 'rb') as f:
            s = struct.Struct('i i i')
            num_docs, num_terms, num_nnz = s.unpack(f.read(s.size))

        return num_docs, num_terms, num_nnz

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
            s_doc_id = struct.Struct('i i')
            s_doc_term = struct.Struct('i f')

            for _ in range(self.num_docs):
                docid, doc_length = s_doc_id.unpack(f.read(s_doc_id.size))
                document = []

                for _ in range(doc_length):
                    termid, value = s_doc_term.unpack(f.read(s_doc_term.size))

                    if not self.transposed:
                        termid, docid = docid, termid

                    document.append((termid, value,))  # add another field to the current document

                yield document


class MmReader2(object):
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
        s = struct.Struct('i i i')
        num_docs, num_terms, num_nnz = s.unpack(input_file.read(s.size))

    def read_headers(self):
        """Reader header row for file metadata

        Returns
        ----------
        num_docs : int
        num_terms : int
        num_nnz : int

        """

        with utils.smart_open(self.input, 'rb') as f:
            s = struct.Struct('i i i')
            num_docs, num_terms, num_nnz = s.unpack(f.read(s.size))

        return num_docs, num_terms, num_nnz

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
            s_docid = struct.Struct('i i')

            for _ in range(self.num_docs):
                docid, doc_length = s_docid.unpack(f.read(s_docid.size))

                s_doc_terms = struct.Struct('i f' * doc_length)
                doc_terms = iter(s_doc_terms.unpack(f.read(s_doc_terms.size)))
                document = []
                for termid in doc_terms:
                    value = next(doc_terms)

                    if not self.transposed:
                        termid, docid = docid, termid

                    document.append((termid, value,))  # add another field to the current document

                yield document
