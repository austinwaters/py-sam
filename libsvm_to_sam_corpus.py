from argparse import ArgumentParser
from itertools import chain
import numpy as np
import re
import sys

from io.corpus import CorpusWriter

from math_util import l2_normalize


def load_libsvm(filename):
    result = {}
    for line in open(filename):
        tokens = re.split('\s+', line.strip())
        if len(tokens) == 0:
            continue

        name = tokens[0]
        feature_counts = {}
        for s in tokens[1:]:
            feature_id, count = s.split(':')
            feature_counts[feature_id] = int(count)
        result[name] = feature_counts
    return result




def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = ArgumentParser()
    parser.add_argument('input_file', type=str, help='libsvm input format')
    parser.add_argument('output_file', type=str, help='Path to destination corpus file')
    options = parser.parse_args(argv[1:])

    instance_dict = load_libsvm(options.input_file)
    num_docs = len(instance_dict)
    feature_ids = sorted(set(chain(*[each.iterkeys() for each in instance_dict.values()])))
    vocab_size = len(feature_ids)
    print 'Read %d docs (vocabulary size %d) from %s' % (num_docs, vocab_size, options.input_file)

    print 'Writing L2-normalized corpus to %s' % options.output_file
    writer = CorpusWriter(options.output_file, data_series='sam', dim=vocab_size)

    # Maps feature_id => dense feature index
    feature_index = {k:i for i, k in enumerate(feature_ids)}

    for name, sparse_features in instance_dict.iteritems():
        # Convert sparse features to dense L2-normalized feature vector
        doc_data = np.zeros((vocab_size, 1))
        for id, count in sparse_features.iteritems():
            doc_data[feature_index[id]] = count
        doc_data = l2_normalize(doc_data)

        writer.write_doc(doc_data, name=name)
    writer.close()


if __name__ == '__main__':
    main(sys.argv)