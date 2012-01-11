from argparse import ArgumentParser
import os
import sys

from condor.condorizable import Condorizable
from io.corpus import CorpusReader
from vem.model import VEMModel


class VEMTask(Condorizable):
    binary = Condorizable.path_to_script(__file__)

    def check_args(self, argv):
        parser = ArgumentParser()
        parser.add_argument('--model', type=str, required=True, help='Save model to <path>, or resume running from that state')
        parser.add_argument('--corpus', type=str, help='Path to SAM corpus')

        parser.add_argument('-T', '--T', type=int, default=10, help='Number of topics')
        parser.add_argument('--iterations', type=int, default=500, help='Run VEM for <n> iterations')
        parser.add_argument('--write_topic_weights', type=str, help='Write topic weights to <path>')
        options = parser.parse_args(argv[1:])

        # If the model doesn't already exist (we're creating a new one), we need to know where the corpus lives
        if not os.path.exists(options.model):
            if options.corpus is None:
                parser.error('Must provide --corpus when creating a new model')
            if not os.path.exists(options.corpus):
                parser.error('Corpus file %s does not exist!' % options.corpus)

        self.add_output_file(options.model)
        return options

    def run(self, options):
        if os.path.exists(options.model):
            print 'Loading model snapshot from %s' % options.model
            model = VEMModel.load(options.model)
        else:
            # Initialize a model from scratch
            print 'Initializing new model'
            reader = CorpusReader(options.corpus, data_series='sam')
            model = VEMModel(reader=reader, T=options.T)

        while model.iteration < options.iterations:
            print '** Iteration %d **' % model.iteration
            model.run_one_iteration()

        if options.write_topic_weights:
            print 'Writing topic weights to %s' % options.write_topic_weights
            with open(options.write_topic_weights, 'w') as f:
                model.write_topic_weights_arff(f)
        model.save(options.model)


if __name__ == '__main__':
    VEMTask(sys.argv)