from argparse import ArgumentParser
from itertools import chain
import numpy as np
import re
import sys

from io.corpus import CorpusWriter
from io.libsvm import load_libsvm

from math_util import l2_normalize


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

    wordlist_path = options.output_file + '.wordlist'
    print 'Writing wordlist to %s' % wordlist_path
    with open(wordlist_path, 'w') as f:
        f.writelines([s + '\n' for s in feature_ids])

if __name__ == '__main__':
    main(sys.argv)