from argparse import ArgumentParser
import sys

from corpus import labelers
from corpus.corpus import CorpusWriter
from vision.gist import grayscale_gist
from math_util import l2_normalize


def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = ArgumentParser()
    parser.add_argument('file_list', type=str, help='File containing list of images to process')
    parser.add_argument('dest_corpus', type=str, help='Path to write GIST corpus')
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


    # Wait to instantiate the corpus writer until we know the dimensionality of the descriptors we'll be writing
    writer = None
    print 'Writing SAM corpus to %s' % options.dest_corpus

    filenames = open(options.file_list).readlines()
    for i, filename in enumerate(filenames):
        filename = filename.strip()
        print 'Processing image %d/%d' % (i+1, len(filenames))

        descriptor = grayscale_gist(filename)
        if writer is None:
            dim = descriptor.size
            writer = CorpusWriter(options.dest_corpus, data_series='sam', dim=dim)

        normalized_descriptor = l2_normalize(descriptor)
        doc_label = labeler(filename) if labeler else None
        writer.write_doc(normalized_descriptor, name=filename, label=doc_label)

    writer.close()


if __name__ == '__main__':
    main(sys.argv)