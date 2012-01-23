from argparse import ArgumentParser
import os
import subprocess
import sys
from condor.condorizable import Condorizable

JAR_FILENAME = 'cross-validator.jar'


class CrossValidationTask(Condorizable):
    """
    Python wrapper around weka cross-validation.

    Example:
        cross_validator_cli.py 'weka.classifiers.lazy.IBk -k 10' foo.arff foo.results
    runs cross-validation of a k-nearest neighbors classifier on the dataset foo.arff.  The mean accuracy and
    confidence interval are written to foo.results.

    """
    binary = 'cv.py'

    def check_args(self, argv):
        parser = ArgumentParser()
        parser.add_argument('--classifier', type=str, required=True, help='Classifier to run')
        parser.add_argument('--flags', type=str, help='(optional) classifier flags')
        parser.add_argument('--data', type=str, required=True, help='Path to arff dataset')
        parser.add_argument('--results', type=str, required=True, help='Save results to <path>')

        options = parser.parse_args(argv[1:])
        if not os.path.exists(options.data):
            parser.error('Data file %s does not exist!' % options.data)
        return options

    def find_cross_validation_jar_or_die(self):
        path = os.path.join(os.path.dirname(self.binary), JAR_FILENAME)
        if os.path.isfile(path):
            return path
        else:
            raise Exception('Cannot locate %s; aborting' % JAR_FILENAME)

    def run(self, options):
        cv_jar_path = self.find_cross_validation_jar_or_die()
        classifier_flags = [] if options.flags is None else options.flags.split()
        command = ['java', '-jar', cv_jar_path, '-classifier', options.classifier, '-data', options.data] \
                  + classifier_flags
        output = subprocess.check_output(command)
        with open(options.results, 'w') as f:
            f.write(output)


if __name__ == '__main__':
    CrossValidationTask(sys.argv)



