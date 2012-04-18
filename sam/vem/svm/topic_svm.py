from argparse import ArgumentParser
import numpy as np
from sklearn.svm import SVC
import sys

from math_util import asrowvector, l2_normalize
from dataset import DataSet
from vem.model import VEMModel
from loss import ClassificationError
import sam.log as log


class TopicSVM(object):
    def __init__(self, sam_model, C=1.0, normalize=False):
        """
        Parameters:
          sam_model: A SAM VEM model to get the topic weights from
          C: SVM margin-accuracy tradeoff, scaled to the number of examples.  Really big values (e.g. 200000) seem to
              work well.
          normalize: If true, use normalized cosine similarity between smoothed document tf in the kernel.
        """
        self.svm = SVC(kernel='precomputed', C=C, scale_C=True)
        self.sam_model = sam_model
        self.topic_transform = np.dot(sam_model.vmu.T, sam_model.vmu)
        if self.topic_transform.shape != (sam_model.T, sam_model.T):
            raise ValueError('Topic transform has shape %s; should be %d squared' %
                             (self.topic_transform.shape, sam_model.T))
        self.normalize = normalize

    def make_gram_matrix(self, left, right=None):
        # left, right: one document per column
        if right is None:
            right = left

        if not self.normalize:
            return np.dot(left.T, np.dot(self.topic_transform, right))
        # else
        left_smoothed = l2_normalize(np.dot(self.sam_model.vmu, left))
        right_smoothed = l2_normalize(np.dot(self.sam_model.vmu, right))
        return np.dot(left_smoothed.T, right_smoothed)

    def train(self, dataset):
        gram_matrix = self.make_gram_matrix(dataset.examples.T)
        self.train_examples = np.copy(dataset.examples.T)
        self.svm.fit(gram_matrix, dataset.targets)

    def predict(self, dataset):
        test_examples = dataset.examples.T
        gram_matrix = self.make_gram_matrix(test_examples, self.train_examples)
        return self.svm.predict(gram_matrix)


def make_dataset(model):
    """ Make a DataSet of inferred topic weights from the documents in the VEM model. """
    mean_topic_weights = model.valpha / asrowvector(np.sum(model.valpha, axis=0))
    targets = model.reader.labels
    keys = np.asarray(model.reader.names, dtype=str)

    dataset = DataSet(mean_topic_weights.T, targets, None, keys=keys)
    return dataset


def run(argv):
    parser = ArgumentParser()
    parser.add_argument('vem_model', type=str, help='SAM VEM model to use features from')
    parser.add_argument('-c', type=float, default=1.0, help='SVM C parameter')
    options = parser.parse_args(argv[1:])

    log.info('Loading SAM model %s' % options.vem_model)

    sam_model = VEMModel.load(options.vem_model)
    log.info('Making dataset')
    dataset = make_dataset(sam_model)

    metric = ClassificationError()
    scores = []
    for i in range(20):
        train_data, test_data = dataset.split(p=0.90, seed=i)

        topic_svm = TopicSVM(sam_model, C=options.c, normalize=True)
        topic_svm.train(train_data)

        predictions = topic_svm.predict(test_data)
        score = metric(test_data.targets, predictions)
        log.info(score)
        scores.append(score)
    log.info('Mean classification error: %g' % np.mean(scores))


if __name__ == '__main__':
    run(sys.argv)
