# Copyright (C) 2018 Radim Rehurek <radimrehurek@seznam.cz>
# cython: embedsignature=True
"""Reader for corpus in the Matrix Market format.

These classes use cython to read/write binary data file

negatives:
  - code is not portable (no check for big vs. little endian)
  - code uses c's fopen/fclose for file access, so won't work with
    non-local files
  - code doesn't use a buffered file stream

"""
from __future__ import with_statement

import logging
import numpy as np

cimport cython

from libc.stdio cimport FILE, fopen, fclose, fwrite, fread, fseek, SEEK_SET
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free


logger = logging.getLogger(__name__)


# hold single termid, value pair
cdef struct TermCount:
    int termid
    float value


cdef class DocumentIterator(object):
    """
    Custom document iterator to avoid duplicating data

    """

    cdef int docid
    cdef TermCount *tc_array
    cdef int doc_length
    cdef int current_index
    cdef bint transposed

    def __cinit__(self, int docid, int doc_length, bint transposed):
        """
        Create iterator by reading doc_length records from file

        """
        self.docid = docid
        self.tc_array = NULL
        self.doc_length = doc_length
        self.current_index = 0
        self.transposed = transposed

    def __iter__(self):
        return self

    def __next__(self):
        cdef TermCount tc
        cdef int termid, temp
        cdef float value

        if self.current_index < self.doc_length:
            tc = self.tc_array[self.current_index]
            self.current_index += 1
            termid = tc.termid
            value = tc.value

            if not self.transposed:
                temp = self.docid
                docid = termid
                termid = temp

            return (termid, value)
        else:
            raise StopIteration

    def __dealloc__(self):
        if self.tc_array is not NULL:
            PyMem_Free(self.tc_array)

cdef object create_DocumentIterator(FILE *file, int docid, int doc_length, bint transposed):
    doc = DocumentIterator(docid, doc_length, transposed)
    doc.tc_array = <TermCount *> PyMem_Malloc(doc_length * sizeof(TermCount))
    fread(doc.tc_array, sizeof(TermCount), doc_length, file)
    return doc


cdef class MmReaderStructAtATime(object):
    """Matrix market file reader (fast Cython version), used for :class:`~gensim.corpora.mmcorpus.MmCorpus`.

    Wrap a term-document matrix on disk (in matrix-market format), and present it
    as an object which supports iteration over the rows (~documents).

    This version reads a single struct at a time from the binary file.  Benefit is not having
    to worry about memory management

    Attributes
    ----------
    num_docs : int
        Number of documents in market matrix file.
    num_terms : int
        Number of terms.
    num_nnz : int
        Number of non-zero terms.

    Notes
    ----------
    Note that the file is read into memory one document at a time, not the whole
    matrix at once (unlike scipy.io.mmread). This allows us to process corpora
    which are larger than the available RAM.

    """
    cdef public input
    cdef public bint transposed
    cdef public int num_docs, num_terms, num_nnz

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
        logger.info("initializing cython corpus reader from %s", input)
        input = input.encode('utf-8')
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

    cdef skip_headers(self, FILE *file):
        """Skip file headers that appear before the first document.

        Parameters
        ----------
        input_file : iterable of str
            Iterable taken from file in MM format.

        """
        cdef int header_value;

        fseek(file, sizeof(header_value) * 3, SEEK_SET)

    def read_headers(self):
        """Reader header row for file metadata

        Returns
        ----------
        num_docs : int
        num_terms : int
        num_nnz : int

        """
        cdef FILE *file
        cdef int num_docs, num_terms, num_nnz

        file = fopen(self.input, "rb")
        fread( & num_docs, sizeof(num_docs), 1, file)
        fread( & num_terms, sizeof(num_terms), 1, file)
        fread( & num_nnz, sizeof(num_nnz), 1, file)
        fclose(file)

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
        cdef FILE *file
        cdef int docid, termid, doc_length
        cdef float value
        cdef TermCount tc

        file = fopen(self.input, "rb")
        self.skip_headers(file)

        while (fread(&docid, sizeof(docid), 1, file) == 1):
            document = []
            fread( & doc_length, sizeof(doc_length), 1, file)

            for i in range(doc_length):
                fread( &tc, sizeof(tc), 1, file)

                if not self.transposed:
                    tc.termid, docid = docid, tc.termid

                document.append((tc.termid, tc.value,))  # add another field to the current document
            yield document

        fclose(file)


cdef class MmReaderStructArray(object):
    """Matrix market file reader (fast Cython version), used for :class:`~gensim.corpora.mmcorpus.MmCorpus`.

    Reads data one document at a time (array of structs)

    Wrap a term-document matrix on disk (in matrix-market format), and present it
    as an object which supports iteration over the rows (~documents).

    Attributes
    ----------
    num_docs : int
        Number of documents in market matrix file.
    num_terms : int
        Number of terms.
    num_nnz : int
        Number of non-zero terms.

    Notes
    ----------
    Note that the file is read into memory one document at a time, not the whole
    matrix at once (unlike scipy.io.mmread). This allows us to process corpora
    which are larger than the available RAM.

    """
    cdef public input
    cdef public bint transposed
    cdef public int num_docs, num_terms, num_nnz

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
        logger.info("initializing cython corpus reader from %s", input)
        input = input.encode('utf-8')
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

    @staticmethod
    def save_corpus(fname, corpus):
        cdef FILE *file
        cdef int termid, docid, doc_length
        cdef int num_docs, num_terms, num_nnz
        cdef float value
        cdef TermCount tc
        logger.info("storing corpus in Matrix Market format to %s", fname)

        file = fopen(fname.encode('utf-8'), "wb")

        # write out header info
        num_docs, num_terms, num_nnz = corpus.num_docs, corpus.num_terms, corpus.num_nnz
        fwrite( & num_docs, sizeof(num_docs), 1, file)
        fwrite( & num_terms, sizeof(num_terms), 1, file)
        fwrite( & num_nnz, sizeof(num_nnz), 1, file)

        for (docid, doc) in enumerate(corpus):
            doc_length = len(doc)
            fwrite( & docid, sizeof(docid), 1, file)
            fwrite( & doc_length, sizeof(doc_length), 1, file)

            for (termid, value) in doc:
                tc.termid, tc.value = termid, value
                fwrite( &tc, sizeof(tc), 1, file)

        fclose(file)

    cdef skip_headers(self, FILE *file):
        """Skip file headers that appear before the first document.

        Parameters
        ----------
        input_file : iterable of str
            Iterable taken from file in MM format.

        """
        cdef int header_value;

        fseek(file, sizeof(header_value) * 3, SEEK_SET)

    def read_headers(self):
        """Reader header row for file metadata

        Returns
        ----------
        num_docs : int
        num_terms : int
        num_nnz : int

        """
        cdef FILE *file
        cdef int num_docs, num_terms, num_nnz

        file = fopen(self.input, "rb")
        fread( & num_docs, sizeof(num_docs), 1, file)
        fread( & num_terms, sizeof(num_terms), 1, file)
        fread( & num_nnz, sizeof(num_nnz), 1, file)
        fclose(file)

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
        cdef FILE *file
        cdef int docid, termid, doc_length
        cdef float value
        cdef TermCount *tc_array
        cdef int temp
        cdef list document
        cdef tuple pair

        file = fopen(self.input, "rb")
        self.skip_headers(file)

        while (fread(&docid, sizeof(docid), 1, file) == 1):
            document = []
            fread( & doc_length, sizeof(doc_length), 1, file)
            tc_array = <TermCount *> PyMem_Malloc(doc_length * sizeof(TermCount))

            fread( tc_array, sizeof(TermCount), doc_length, file)

            for i in range(doc_length):
                termid = tc_array[i].termid
                value = tc_array[i].value

                if not self.transposed:
                    temp = docid
                    docid = termid
                    termid = temp

                pair = (termid, value)
                document.append(pair)  # add another field to the current document

            PyMem_Free(tc_array)
            yield document

        fclose(file)


cdef class MmReaderStructArrayDocumentIterator(object):
    """Matrix market file reader (fast Cython version), used for :class:`~gensim.corpora.mmcorpus.MmCorpus`.

    Reads data one document at a time (array of structs) into a numpy array.  Hope is that
    using numpy arrays lets us allocate memory for each document's array at once (since
    we know the size of the doc) instead of using a python list where we start with an
    empty list and append each doc one at at time

    Wrap a term-document matrix on disk (in matrix-market format), and present it
    as an object which supports iteration over the rows (~documents).

    Attributes
    ----------
    num_docs : int
        Number of documents in market matrix file.
    num_terms : int
        Number of terms.
    num_nnz : int
        Number of non-zero terms.

    Notes
    ----------
    Note that the file is read into memory one document at a time, not the whole
    matrix at once (unlike scipy.io.mmread). This allows us to process corpora
    which are larger than the available RAM.

    """
    cdef public input
    cdef public bint transposed
    cdef public int num_docs, num_terms, num_nnz

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
        logger.info("initializing cython corpus reader from %s", input)
        input = input.encode('utf-8')
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

    cdef skip_headers(self, FILE *file):
        """Skip file headers that appear before the first document.

        Parameters
        ----------
        input_file : iterable of str
            Iterable taken from file in MM format.

        """
        cdef int header_value;

        fseek(file, sizeof(header_value) * 3, SEEK_SET)

    def read_headers(self):
        """Reader header row for file metadata

        Returns
        ----------
        num_docs : int
        num_terms : int
        num_nnz : int

        """
        cdef FILE *file
        cdef int num_docs, num_terms, num_nnz

        file = fopen(self.input, "rb")
        fread( & num_docs, sizeof(num_docs), 1, file)
        fread( & num_terms, sizeof(num_terms), 1, file)
        fread( & num_nnz, sizeof(num_nnz), 1, file)
        fclose(file)

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
        cdef FILE *file
        cdef int docid, doc_length

        file = fopen(self.input, "rb")
        self.skip_headers(file)

        while (fread(&docid, sizeof(docid), 1, file) == 1):
            fread( & doc_length, sizeof(doc_length), 1, file)
            document = create_DocumentIterator(file, docid, doc_length, self.transposed)
            yield document

        fclose(file)


cdef class MmReaderStructArrayReadOnly(object):
    """Matrix market file reader (fast Cython version), used for :class:`~gensim.corpora.mmcorpus.MmCorpus`.

    This version simply reads the binary data but does not return it.  Meant to benchmark how
    much time the IO itself is taking vs. creating python return objects

    Wrap a term-document matrix on disk (in matrix-market format), and present it
    as an object which supports iteration over the rows (~documents).

    Attributes
    ----------
    num_docs : int
        Number of documents in market matrix file.
    num_terms : int
        Number of terms.
    num_nnz : int
        Number of non-zero terms.

    Notes
    ----------
    Note that the file is read into memory one document at a time, not the whole
    matrix at once (unlike scipy.io.mmread). This allows us to process corpora
    which are larger than the available RAM.

    """
    cdef public input
    cdef public bint transposed
    cdef public int num_docs, num_terms, num_nnz

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
        logger.info("initializing cython corpus reader from %s", input)
        input = input.encode('utf-8')
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

    cdef skip_headers(self, FILE *file):
        """Skip file headers that appear before the first document.

        Parameters
        ----------
        input_file : iterable of str
            Iterable taken from file in MM format.

        """
        cdef int header_value;

        fseek(file, sizeof(header_value) * 3, SEEK_SET)

    def read_headers(self):
        """Reader header row for file metadata

        Returns
        ----------
        num_docs : int
        num_terms : int
        num_nnz : int

        """
        cdef FILE *file
        cdef int num_docs, num_terms, num_nnz

        file = fopen(self.input, "rb")
        fread( & num_docs, sizeof(num_docs), 1, file)
        fread( & num_terms, sizeof(num_terms), 1, file)
        fread( & num_nnz, sizeof(num_nnz), 1, file)
        fclose(file)

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
        cdef FILE *file
        cdef int docid, termid, doc_length
        cdef float value
        cdef TermCount *tc_array
        cdef int temp

        file = fopen(self.input, "rb")
        self.skip_headers(file)

        while (fread(&docid, sizeof(docid), 1, file) == 1):
            document = []
            fread( & doc_length, sizeof(doc_length), 1, file)
            tc_array = <TermCount *> PyMem_Malloc(doc_length * sizeof(TermCount))

            fread(tc_array, sizeof(TermCount), doc_length, file)

            for i in range(doc_length):
                termid = tc_array[i].termid
                value = tc_array[i].value

                if not self.transposed:
                    temp = docid
                    docid = termid
                    termid = temp

                #document.append((termid, value,))  # add another field to the current document

            PyMem_Free(tc_array)
            yield document

        fclose(file)