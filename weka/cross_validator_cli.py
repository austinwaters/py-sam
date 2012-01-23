from argparse import ArgumentParser
import os
import subprocess
import sys
from condor.condorizable import Condorizable

# Path to the jar relative to the binary
CV_JAR_PATH = 'weka/cross-validator.jar'


class CrossValidationTask(Condorizable):
    """
    Python wrapper around weka cross-validation.  This wrapper is Condorizable, so that CV jobs can be started on condor
    both from the command-line and programatically through python (by calling CrossValidationTask(kw=...)).

    Example:
        cross_validator_cli.py --classifier weka.classifiers.lazy.IBk --flags '-k 10' --data foo.arff --results foo.results
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
        self.add_output_file(options.results)
        return options

    def find_cross_validation_jar_or_die(self):
        path = os.path.join(os.path.dirname(self.binary), CV_JAR_PATH)
        if os.path.isfile(path):
            return path
        else:
            raise Exception('Cannot locate %s; aborting' % CV_JAR_PATH)

    def run(self, options):
        cv_jar_path = self.find_cross_validation_jar_or_die()
        classifier_flags = [] if options.flags is None else options.flags.split()
        command = ['java', '-jar', cv_jar_path, '-classifier', options.classifier, '-data', options.data] \
                  + classifier_flags
        output = subprocess.check_output(command)
        with open(options.results, 'w') as f:
            f.write(output)


def run_cv_batch(cv_configs):
    for job_settings in cv_configs:
        results_file = job_settings['results']
        if os.path.exists(results_file):
            print 'Warning: results file %s already exists; aborting' % results_file
            continue
        if Condorizable.is_locked(results_file):
            print 'WARNING: Results file %s is locked; check that another job isn''t writing to this path' %\
                  results_file
            continue
        CrossValidationTask(kw=job_settings)


