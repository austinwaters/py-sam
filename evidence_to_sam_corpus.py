from argparse import ArgumentParser
from itertools import chain
import numpy as np
import sys

from io.corpus import CorpusWriter
from io.evidence import load_evidence_file
from io import labelers

from math_util import l2_normalize


def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = ArgumentParser()
    parser.add_argument('input_file', type=str, help='Input file in evidence format')
    parser.add_argument('output_file', type=str, help='Path to destination corpus file')
    parser.add_argument('--labeler', type=str, help='Labeler to apply')
    options = parser.parse_args(argv[1:])

    labeler = None
    if options.labeler is None:
        print 'Warning: no labeler provided'
    elif options.labeler not in labelers.registry:
        labeler_names = ', '.join(sorted(labelers.registry.keys()))
        parser.error('Invalid labeler "%s"; available options are %s' % (options.labeler, labeler_names))
    else:
        labeler = labelers.registry[options.labeler]

    instance_dict = load_evidence_file(options.input_file)
    num_docs = len(instance_dict)
    feature_ids = sorted(set(chain(*[each.iterkeys() for each in instance_dict.values()])))
    vocab_size = len(feature_ids)
    print 'Read %d docs (vocabulary size %d) from %s' % (num_docs, vocab_size, options.input_file)

    print 'Writing L2-normalized corpus to %s' % options.output_file
    writer = CorpusWriter(options.output_file, data_series='sam', dim=vocab_size)

    # Create a map of feature_id => dense feature index
    feature_index = {k:i for i, k in enumerate(feature_ids)}

    # For each document, convert sparse features to dense L2-normalized feature vector and write it to the corpus
    for name, sparse_features in instance_dict.iteritems():
        doc_data = np.zeros((vocab_size, 1))
        for id, count in sparse_features.iteritems():
            doc_data[feature_index[id]] = count
        doc_data = l2_normalize(doc_data)
        doc_label = labeler(name) if labeler else None

        writer.write_doc(doc_data, name=name, label=doc_label)
    writer.close()

    wordlist_path = options.output_file + '.wordlist'
    print 'Writing wordlist to %s' % wordlist_path
    with open(wordlist_path, 'w') as f:
        f.writelines([s + '\n' for s in feature_ids])

if __name__ == '__main__':
    main(sys.argv)