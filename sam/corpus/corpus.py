from itertools import chain
import os
from math import floor
import h5py
from numpy.random import shuffle
import numpy as np
from sam.corpus.evidence import load_evidence_file
from sam.math_util import asvector, sum_lt, numpy_random_seed_temporarily, l2_normalize

NAMES_SERIES = 'names'
LABELS_SERIES = 'labels'


class EvidenceReader:
    """
    Interface for reading evidence files.
    """
    def __init__(self, filename, labeler=None):
        self.filename = filename

        instance_dict = load_evidence_file(filename)
        num_docs = len(instance_dict)
        doc_names = sorted(instance_dict.keys())
        feature_ids = sorted(set(chain(*[each.iterkeys() for each in instance_dict.values()])))
        vocab_size = len(feature_ids)

        # Create a map of feature_id => dense feature index
        feature_index = {k:i for i, k in enumerate(feature_ids)}

        doc_matrix = np.zeros((num_docs, vocab_size))
        doc_labels = []

        # For each document, convert sparse features to dense L2-normalized feature vector and write it into the
        # document matrix
        for i, name in enumerate(doc_names):
            sparse_features = instance_dict[name]

            doc_data = np.zeros(vocab_size)
            for id, count in sparse_features.iteritems():
                doc_data[feature_index[id]] = count
            doc_data = l2_normalize(doc_data)
            doc_labels[i] = labeler(name) if labeler else None
            doc_matrix[i, :] = doc_data

        self.num_docs = num_docs
        self.raw_labels = doc_labels
        self.class_names = sorted(set(doc_labels))
        self.doc_matrix = doc_matrix
        self.dim = len(feature_index)

    def read_doc(self, d):
        return self.doc_matrix[d, :]


class CorpusReader:
    """
    Read a document:
    data = reader.read_doc(14)
    doc_label = reader.labels[14]
    doc_name = reader.names[14]
    class_names_of_doc_1 = reader.class_names[reader.labels[1]]
    """
    def __init__(self, filename=None, base_group='data', data_series='sift'):
        self.base_group_name = base_group
        self.data_series_name = data_series
        self.open(filename)
        self.init_labels()

    def open(self, filename):
        self.filename = filename

        if not os.path.isfile(filename):
            raise Exception("Corpus file %s does not exist!" % filename)

        self.f = h5py.File(filename, 'r')
        if self.base_group_name:
            self.hdf_base = self.f[self.base_group_name]
        else:
            self.hdf_base = self.f

        if self.data_series_name not in self.hdf_base:
            raise Exception("File %s does not contain a dataset named /%s/%s" % (filename, self.base_group_name,
                                                                                 self.data_series_name))
        self.data = self.hdf_base[self.data_series_name]
        self.dim, self.num_data = self.data.shape
        
        self.names = self.hdf_base[NAMES_SERIES]
        self.raw_labels = list(self.hdf_base[LABELS_SERIES])
        self.class_names = list(set(self.raw_labels))  # List of unique labels
        self.num_classes = len(self.class_names)

        # Doc index
        self.doc_index = self.hdf_base['doc_index'].value
        if len(self.doc_index.shape) != 2:
            raise Exception("Doc index must be two dimensional")
        if self.doc_index.shape[0] != 2:
            self.doc_index = self.doc_index.T

        self.doc_sizes = self.doc_index[1, :] - self.doc_index[0, :] + 1
        self.num_docs = self.doc_index.shape[1]

        # For get_datum_in_doc
        self.current_doc = None
        self.current_base_index = None
        self.doc_data = None

        # Buffer info for get_datum
        self.range_in_memory = None
        self.buffer = None
        self.buffer_size = 10000

    def init_labels(self):
        """
        Builds the initial set of labels and label map for the corpus.
        """
        self.raw_labels = list(self.hdf_base[LABELS_SERIES])
        self.class_names = sorted(list(set(self.raw_labels)))  # List of unique labels
        self.num_classes = len(self.class_names)
        
        self.labels = np.array([self.class_names.index(each) for each in self.raw_labels], dtype='int8')  # Class indices
        self.labels_observed = np.array([each != -1 for each in self.labels], dtype='bool')

        assert self.num_classes < 128
        assert len(self.labels) == self.num_docs
        assert len(self.labels_observed)== self.num_docs

    def random_stratified_subset(self, p=None, n=None, seed=None):
        """
        Gets a fraction of the documents in each class.  Returns a list of document indices.
        """
        # XXX this only selects documents with observed labels
        if not bool(p) ^ bool(n):
            raise ValueError('Must provide exactly one of p or n kwargs')

        docs_selected = np.zeros((self.num_docs,), dtype='bool')
        docs_in_classes = [np.flatnonzero((self.labels == c) * self.labels_observed) for c in range(self.num_classes)]
        num_docs_in_classes = [len(each) for each in docs_in_classes]

        # Determine number of documents to select per class
        if p is not None:
            # Ensure p is a float between 0 and 1
            p = float(p)
            if p < 0.0 or p >= 1.0:
                raise ValueError("p must be between 0 and 1 (got: %f)" % p)
            if p == 0.0:
                return []
            num_docs_to_select = [int(floor(p * each)) for each in num_docs_in_classes]
        if n is not None:
            n = int(n)
            if n == 0:
                return []
            if n < 0:
                raise ValueError('n must be positive (got: %d)' % n)
            num_docs_to_select = [n] * self.num_classes

        with numpy_random_seed_temporarily(seed):
            for c, docs in enumerate(docs_in_classes):
                shuffle(docs)

                if num_docs_to_select[c] == 0 or num_docs_to_select[c] > num_docs_in_classes[c]:
                    raise Exception("Class %d doesn't have enough labeled documents to select %d (has %d)" % \
                                    (c, num_docs_to_select[c], num_docs_in_classes[c]))
                docs_selected[docs[0:num_docs_to_select[c]]] = True
        return np.flatnonzero(docs_selected)

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Try to locate the corpus file if it doesn't can't be found in the last location
        if not os.path.exists(self.filename):
            try:
                self.filename = find_corpus(os.path.basename(self.filename))
            except Exception:
                # TODO: HACK -- Sometimes, e.g. when collecting results, we just want to load the model but don't
                # actually need to read from the corpus.  This should probably be handled by creating a custom
                # unpickler or something...
                pass

        if os.path.isfile(self.filename):
            self.open(self.filename)
        else:
            print 'Warning: unable to find corpus %s' % state['filename']

    def __getstate__(self):
        return {
            'base_group_name':self.base_group_name,
            'data_series_name':self.data_series_name,
            'filename':self.filename,
            'labels':self.labels,
            'labels_observed':self.labels_observed,
        }

    def read_doc(self, d):
        if d < 0 or d >= self.num_docs:
            raise Exception("Document index %d is out of bounds (num docs: %d)" % (d, self.num_docs))
        begin_index, end_index = self.doc_index[:, d]
        return self.data[:, begin_index:end_index+1]

    def get_datum_in_doc(self, doc, i):
        # Read the doc for i, if it's not already in memory
        # Get it out of the doc, return it as a column vector
        if self.current_doc != doc:
            self.current_doc = doc
            self.current_base_index = self.doc_index[doc, 0]
            self.doc_data = self.read_doc(doc)
        return asvector(self.doc_data[:, i-self.current_base_index])

    def get_datum(self, i, buffered=True):
        if buffered:
            return self.get_datum_buffered(i)
        else:
            return self.data[:, i]

    def get_datum_buffered(self, i):
        if self.range_in_memory is None or i < self.range_in_memory[0] or i >= self.range_in_memory[1]:
            # buffer_size = 10000, i = 20042
            # ==> range_in_memory = 20000, 30000, index_in_buffer = 42
            buffer_start = i - i % self.buffer_size
            buffer_end = min(buffer_start + self.buffer_size, self.num_data)
            self.buffer = self.data[:, buffer_start:buffer_end]
            self.range_in_memory = (buffer_start, buffer_end)
        return self.buffer[:, i-self.range_in_memory[0]]

    
class CorpusWriter:
    def __init__(self, filename, base_group='data', data_series='sift', compression=True, dim=128):
        self.filename = filename
        self.base_group = base_group
        self.data_series = data_series
        self.dim = dim
        self.compression = compression

        self.f = h5py.File(filename, 'w')
        self.group = self.f.create_group(base_group)
        if compression:
            self.dataset = self.group.create_dataset(data_series, shape=(dim, 0), maxshape=(dim, None),
                                                     dtype='float', chunks=True, compression='gzip')
        else:
            self.dataset = self.group.create_dataset(data_series, shape=(dim, 0), maxshape=(dim, None),
                                                     dtype='float', chunks=True)
        vlen_string_type = h5py.special_dtype(vlen=str)
        self.names = self.group.create_dataset(NAMES_SERIES, shape=(0,), maxshape=(None,), dtype=vlen_string_type)
        self.labels = self.group.create_dataset(LABELS_SERIES, shape=(0,), maxshape=(None,), dtype=vlen_string_type)
        self.num_data = 0
        self.num_docs = 0
        self.doc_sizes = []

    def __del__(self):
        if self.f:
            self.close()

    def close(self):
        self.write_doc_index()
        self.f.close()
        self.f = None

    def write_doc(self, data, name=None, label=None):
        name = name or ''
        label = label or ''
        if type(label) != str:
            raise ValueError("Label must be a string (got: %s)" % str(label))

        if data.ndim != 2:
            raise ValueError("Data array should be two-dimensional (has ndim=%d)" % data.ndim)

        dim, doc_size = data.shape
        if dim != self.dim:
            raise Exception("Document data has incompatible dimensionality (%d, should be %d)" % (dim, self.dim))
        self.num_data += doc_size
        self.num_docs += 1
        self.doc_sizes.append(doc_size)
        self.dataset.resize((dim, self.num_data))
        self.dataset[:, -doc_size:] = data
        self.names.resize((self.num_docs,))
        self.names[-1] = name
        self.labels.resize((self.num_docs,))
        self.labels[-1] = label

    def write_doc_index(self):
        doc_index_dataset = self.group.create_dataset('doc_index', shape=(2, self.num_docs), dtype='int32')
        # e.g. docsizes 10, 25 -> [0, 10; 9, 34]
        doc_index_dataset[0, :] = sum_lt(np.array(self.doc_sizes))
        doc_index_dataset[1, :] = np.cumsum(np.array(self.doc_sizes)) - 1


def corpus_mean(reader):
    sum = np.zeros((reader.dim,), 'float64')
    for d in range(reader.num_docs):
        data = reader.read_doc(d)
        np.add(sum, data.sum(axis=1), out=sum)
    return asvector(sum / reader.num_data)

def corpus_mean_and_precision(reader):
    sum = np.zeros((reader.dim,), 'float64')
    sum_squared_norms = 0.
    
    for d in range(reader.num_docs):
        data = reader.read_doc(d)
        np.add(sum, data.sum(axis=1), out=sum)
        sum_squared_norms += (data ** 2).sum()

    mean = sum / reader.num_data
    squared_norm_mean = (mean ** 2).sum()
    return mean, reader.dim * reader.num_data / (sum_squared_norms - 2*mean.dot(sum) + reader.num_data * squared_norm_mean)


search_path = [os.getcwd(),  # Current directory
               os.path.join(os.getenv('HOME'), 'data'),   # ~/data/
               '/projects/nn/austin/research/data'  # Other places...
               ]

def find_corpus(name):
    if os.path.isabs(name):
        if not os.path.isfile(name):
            raise Exception("Cannot find corpus %s!" % name)
        return name
    else:
        for dir in search_path:
            full_path = os.path.abspath(os.path.join(dir, name))  # search_path may be relative
            if os.path.isfile(full_path):
                return full_path
        raise Exception("Cannot find corpus %s!" % name)

# TESTS
def writer_basic_test():
    """
    Try just writing some documents to a corpus.
    """
    import tempfile
    filename = tempfile.mktemp()
    writer = CorpusWriter(filename)
    try:
        for i in range(1, 11):
            data = np.ones((128, 100)) * i
            writer.write_doc(data)
        writer.close()
    finally:
        os.remove(filename)

def read_write_test():
    """
    Write some documents, then make sure they can be read back correctly.
    """
    import tempfile
    filename = tempfile.mktemp()
    try:
        dim = 42
        doc_sizes = [27, 4, 8]
        doc_contents = [4.2, 0.12, 1.0]

        # Write
        writer = CorpusWriter(filename, dim=dim)
        for size, contents in zip(doc_sizes, doc_contents):
            data = np.ones((dim, size)) * contents
            writer.write_doc(data)
        writer.close()

        # Read
        reader = CorpusReader(filename)
        assert reader.num_docs == len(doc_sizes)
        assert reader.num_data == sum(doc_sizes)

        # Try reading whole documents
        for i in range(reader.num_docs):
            data = reader.read_doc(i)
            assert data.shape[0] == dim  # Correct size?
            assert all(data == doc_contents[i])  # Correct contents?

        # Try reading single data points
        datum = reader.get_datum(doc_sizes[0] - 1)
        assert all(datum == doc_contents[0])

        datum = reader.get_datum(doc_sizes[0])
        assert all(datum == doc_contents[1])
    finally:
        os.remove(filename)

    
def read_buffered_test(corpus_filename='different.h5'):
    """ Tests that buffered and unbuffered reads from the corpus return identical data. """
    corpus_filename = find_corpus(corpus_filename)

    import numpy.random
    reader = CorpusReader(corpus_filename)
    random_data_indices = sorted(numpy.random.randint(reader.num_data, size=100))
    for i in random_data_indices:
        unbuffered_result = reader.get_datum(i, buffered=False)
        buffered_result = reader.get_datum(i, buffered=True)
        assert (unbuffered_result == buffered_result).all(), \
            'Reads for %d differ:  %s vs. %s' % (i, unbuffered_result, buffered_result)

    unbuffered_result = reader.get_datum(reader.num_data-1, buffered=False)
    buffered_result = reader.get_datum(reader.num_data-1, buffered=True)
    assert (unbuffered_result == buffered_result).all()

def corpus_precision_test():
    corpus_filename = find_corpus('different.h5')
    reader = CorpusReader(corpus_filename)
    _, actual = corpus_mean_and_precision(reader)  # Ignore returned mean
    expected = 6.650818457547040e-04

    assert np.allclose(actual, expected), '%f vs. %f' % (actual, expected)
