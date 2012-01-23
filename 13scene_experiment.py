from argparse import ArgumentParser
import inspect
from itertools import product
import os
import sys
from condor.condorizable import Condorizable

from vem.cli import VEMTask
from weka.cross_validator_cli import CrossValidationTask


def dict_product(config):
    keys = config.keys()
    values = config.values()
    for each in product(*values):
        d = dict(zip(keys, each))
        for k, v in d.iteritems():
            if inspect.isfunction(v):
                d[k] = v(d)
        yield d


"""
Functions to dynamically generate parts of the config for SAM.  We use the following path conventions:
Model filename:  <corpus>--<topics>T.model
Topic weights filename:  <corpus>--<topics>T.arff
Weka results:  <corpus>--<topics>T--<classifier>.results
"""

def get_topic_weight_filename(config):
    """
    Constructs the topic weights file from the rest of the config.
    """
    base = os.path.splitext(config['corpus'])[0]
    return '%s--%dT.arff' % (base, config['T'])

def get_model_filename(config):
    """
    Constructs the model filename from the rest of the config.
    """
    base = os.path.splitext(config['corpus'])[0]
    return '%s--%dT.model' % (base, config['T'])

vem_config = {
    'model':[get_model_filename],
    'corpus':['13scene-gist.h5', '13scene-color-gist.h5'],
    'T':[10, 20, 30, 40, 50],
    'iterations':[100],
    'write_topic_weights':[get_topic_weight_filename],
    'condor':[''],
}
vem_configs = list(dict_product(vem_config))


def run_sam():
    """
    Runs SAM on every experimental configuration defined by 'config'.  Jobs that have already been run or are
    current running (i.e. for which the model file already exists, or for which a lock file exists) will be skipped.
    """
    for job_settings in dict_product(vem_configs):
        print 'SAM', job_settings

        model_file = job_settings['model']
        if os.path.exists(model_file):
            print 'WARNING: Model %s already exists; skipping' % os.path.basename(model_file)
            continue
        if Condorizable.is_locked(model_file):
            print 'WARNING: Model %s is locked; check that another job isn''t writing to this path' % \
                  os.path.basename(model_file)
            continue

        VEMTask(kw=job_settings)


# Functions for constructing configs for running cross-validation
def get_cv_results_filename(config):
    """
    Constructs the weka results file from the rest of the weka config.
    """
    base = os.path.splitext(config['arff'])[0]
    classifier_parts = config['classifier'].split()
    classifier_class, classifier_options = classifier_parts[0], classifier_parts[1:]
    classifier_name = classifier_class.split('.')[-1]  # e.g. SimpleLogistic, IBk
    options_desc = classifier_options.join()

    # e.g. 13scene-gist.SimpleLogistic-K5.results
    return '%s.%s%s.results' % (base, classifier_name, options_desc)

cv_config = {
    'classifier':['weka.classifiers.lazy.IBk', 'weka.classifiers.functions.SimpleLogistic'],
    'data':[each['write_topic_weights'] for each in vem_configs],
    'results':[get_cv_results_filename],
    'condor':['']
    }
cv_configs = list(dict_product(cv_config))


def run_cv():
    for job_settings in dict_product(cv_configs):
        print 'CV', job_settings
        CrossValidationTask(kw=job_settings)


tasks = {
    'sam':run_sam,
    'weka':run_cv,
}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('task', type=str, required=True, choices=tasks.keys())
    options = parser.parse_args(sys.argv[1:])

    # Run the task
    tasks[options.task]()

