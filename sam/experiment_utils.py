from itertools import product
import inspect
import os


def enumerate_configs(config):
    keys = config.keys()
    values = config.values()
    result = []
    for each in product(*values):
        d = dict(zip(keys, each))
        for k, v in d.iteritems():
            if inspect.isfunction(v):
                d[k] = v(d)
        result.append(d)
    return result


def get_topic_weight_filename(config):
    """
    Constructs the SAM topic weights file from the rest of the config.
    """
    base = os.path.splitext(config['corpus'])[0]
    return '%s--%dT.arff' % (base, config['T'])


def get_model_filename(config):
    """
    Constructs the SAM model filename from the rest of the config.
    """
    base = os.path.splitext(config['corpus'])[0]
    return '%s--%dT.model' % (base, config['T'])


def get_cv_results_filename(config):
    """
    Constructs the weka results file from the rest of the weka config.
    """
    base = os.path.splitext(config['data'])[0]
    classifier = config['classifier'].split('.')[-1]    # e.g. SimpleLogistic, IBk
    classifier_options = config['flags'].replace(' ', '') if 'flags' in config else ''
    # e.g. 13scene-gist.SimpleLogistic-K5.results
    return '%s.%s%s.results' % (base, classifier, classifier_options)
