"""
Writes an ARFF file containing GIST descriptors from a set of images.

Usage:
make_gist_arff.py <file_list> <dest> --labeler <labeler>

Required parameters:
    file_list: a file containing a list of the images to process (one per line)
    dest: where to write the arff file
    labeler: specifies how to map image filename -> label

Optional flags:
    --color: extract color GIST (one GIST per each color channel)  (default: use grayscale GIST)
    --normalize: write L2 normalized GIST descriptors (default: unnormalized)
"""

from argparse import ArgumentParser
from itertools import izip
import os
import sys

from arff import ArffWriter
from corpus import labelers
from vision.gist import color_gist, grayscale_gist
from math_util import l2_normalize
from condor.condorizable import Condorizable


class MakeGistArffTask(Condorizable):
    binary = Condorizable.path_to_script(__file__)

    def check_args(self, argv):
        parser = ArgumentParser()
        parser.add_argument('file_list', type=str, help='File containing list of images to process')
        parser.add_argument('dest', type=str, help='Destination ARFF file')
        parser.add_argument('--labeler', type=str, required=True, choices=labelers.registry.keys(), help='Labeler to apply')
        parser.add_argument('--color', action='store_true', help='Color GIST?')
        parser.add_argument('--normalize', action='store_true', help='L2 normalize GIST data?')
        options = parser.parse_args(argv[1:])

        if not os.path.exists(options.file_list):
            parser.error('Input file %s does not exist!' % options.file_list)

        self.add_output_file(options.dest)
        return options

    def run(self, options):
        labeler = labelers.registry[options.labeler]

        # Wait to instantiate the corpus writer until we know the dimensionality of the descriptors we'll be writing
        filenames = open(options.file_list).readlines()
        labels = [labeler(each) for each in filenames]
        class_list = sorted(set(labels))

        writer = ArffWriter(options.dest, class_list=class_list)
        print 'Writing GIST data to %s' % options.dest

        for i, (filename, label) in enumerate(izip(filenames, labels)):
            filename = filename.strip()
            print 'Processing image %d/%d' % (i+1, len(filenames))

            descriptor = color_gist(filename) if options.color else grayscale_gist(filename)

            if options.normalize:
                descriptor = l2_normalize(descriptor)
            writer.write_example(descriptor, label)
        writer.close()


if __name__ == '__main__':
    MakeGistArffTask(sys.argv)

