"""
Run the full 8scene pipeline:  corpus -> sam topic weights -> classification results

Path conventions:
Model filename:  <corpus>--<topics>T.model
Topic weights filename:  <corpus>--<topics>T.arff
Weka results:  <corpus>--<topics>T--<classifier>.results
"""

from vem.cli import run_sam_batch
from weka.cross_validator_cli import run_cv_batch
from experiment_utils import *


# SAM
vem_config = {
    'model':[get_model_filename],
    'corpus':['8scene-gist.h5', '8scene-color-gist.h5'],
    'T':[10, 20, 30, 40, 50, 60, 70],
    'iterations':[100],
    'write_topic_weights':[get_topic_weight_filename],
    'condor':[''],
    }
vem_configs = enumerate_configs(vem_config)

# K-NN
knn_configs = {
    'classifier':['weka.classifiers.lazy.IBk'],
    'flags':['-K 5', '-K 10', '-K 15'],
    'data':[each['write_topic_weights'] for each in vem_configs],
    'results':[get_cv_results_filename],
    'condor':['']
}
# Logistic regression
lr_configs = {
    'classifier':['weka.classifiers.functions.SimpleLogistic'],
    'data':[each['write_topic_weights'] for each in vem_configs],
    'results':[get_cv_results_filename],
    'condor':['']
}
classifier_configs = enumerate_configs(knn_configs) + enumerate_configs(lr_configs)


if __name__ == '__main__':
    run_sam_batch(vem_configs)
    run_cv_batch(classifier_configs)