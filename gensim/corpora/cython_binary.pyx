# Copyright (C) 2018 Radim Rehurek <radimrehurek@seznam.cz>
# cython: embedsignature=True
"""Reader for corpus in the Matrix Market format."""
from __future__ import with_statement

from gensim import utils

from six import string_types
from six.moves import xrange
import logging

cimport cython
from libc.stdio cimport FILE, fopen, fclose, sscanf, fwrite, fread, fwrite, fprintf, fseek, SEEK_SET



logger = logging.getLogger(__name__)


cdef class MmReader(object):
    """Matrix market file reader (fast Cython version), used for :class:`~gensim.corpora.mmcorpus.MmCorpus`.

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
        self.input, self.transposed = input, transposed
        return
        with utils.open_file(self.input) as lines:
            try:
                header = utils.to_unicode(next(lines)).strip()
                if not header.lower().startswith('%%matrixmarket matrix coordinate real general'):
                    raise ValueError(
                        "File %s not in Matrix Market format with coordinate real general; instead found: \n%s" %
                        (self.input, header)
                    )
            except StopIteration:
                pass

            self.num_docs = self.num_terms = self.num_nnz = 0
            for lineno, line in enumerate(lines):
                line = utils.to_unicode(line)
                if not line.startswith('%'):
                    self.num_docs, self.num_terms, self.num_nnz = (int(x) for x in line.split())
                    if not self.transposed:
                        self.num_docs, self.num_terms = self.num_terms, self.num_docs
                    break

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

    def write_corpus(self, corpus):
        cdef FILE *file
        cdef int termid, docid, doc_length
        cdef int num_docs, num_terms, num_nnz
        cdef float value

        file = fopen(self.input.encode('utf-8'), "wb")

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
                fwrite( & termid, sizeof(termid), 1, file)
                fwrite( & value, sizeof(value), 1, file)

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
        cdef int num_docs, num_terms, num_nnz
        cdef float value

        file = fopen(self.input.encode('utf-8'), "rb")

        self.skip_headers(file)

        while (fread(&docid, sizeof(docid), 1, file) == 1):
            document = []
            fread(&doc_length, sizeof(doc_length), 1, file)

            for i in range(doc_length):
                fread( & termid, sizeof(termid), 1, file)
                fread( & value, sizeof(value), 1, file)

                if not self.transposed:
                    termid, docid = docid, termid

                document.append((termid, value,))  # add another field to the current document
            yield document

        fclose(file)