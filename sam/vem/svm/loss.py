"""
Loss functions.
"""

import numpy as np
import sam.log as log


class SquaredLoss(object):
    def __call__(self, targets, predictions):
        return np.sum((np.asarray(targets) - np.asarray(predictions)) ** 2)


class LogLoss(object):
    def __call__(self, targets, predictions):
        return np.sum(-np.log((predictions ** targets) * (1.0 - predictions) ** (1.0 - targets)))


class MeanSquaredError(object):
    def __call__(self, targets, predictions):
        return np.mean((np.asarray(targets) - np.asarray(predictions)) ** 2)


class ClassificationError(object):
    def __call__(self, targets, predictions):
        if len(targets) != len(predictions):
            raise ValueError('Targets and predictions have different lengths')
        num_incorrect = np.sum(targets != np.round(predictions)) # TODO: generalize to use decision rule object
        return num_incorrect / float(len(targets))


class PrecisionRecallBreakEvenLoss(object):
    """
    A loss based on the precision-recall break even point (the precision, or equivalently recall, at the position
    num_positive_examples).  Returns 1-PRBEP.
    """
    def __call__(self, targets, predictions):
        num_positives = np.count_nonzero(targets == 1.0)

        # Sort the examples by the model predictions
        sorted_indices = np.argsort(predictions)
        predictions = predictions[sorted_indices]
        targets = targets[sorted_indices]

        # Compute precision @ (num positive examples), which also equals the recall
        precision = float(np.count_nonzero(targets[-num_positives:] == 1.0))/num_positives
        return 1.-precision


class ClassificationCost(object):
    def __init__(self, fp_weight=1.0, fn_weight=1.0):
        """
        Parameters:
            fp_weight: Cost of each false positive
            fn_weight: Cost of each false negative
        """
        self.fp_weight = fp_weight
        self.fn_weight = fn_weight

    def __call__(self, targets, predictions):
        predictions = np.round(predictions)  # TODO: generalize to use decision rule object
        num_false_positives = np.sum((targets != 1.0) * (predictions == 1.0))
        num_false_negatives = np.sum((targets == 1.0) * (predictions != 1.0))
        return num_false_positives * self.fp_weight + num_false_negatives * self.fn_weight


class RankingLoss(object):
    """
    What proportion of (pos, neg) example pairs are out of order?
    """
    def __call__(self, targets, predictions):
        n = len(targets)

        # Sort the predictions and targets by the prediction value (increasing)
        sorted_indices = np.argsort(predictions)
        predictions = predictions[sorted_indices]
        targets = targets[sorted_indices]

        # Positive labels are always +1, but negative examples may be 0 or -1
        num_pos = (targets == 1.0).sum()
        num_neg = (targets != 1.0).sum()

        num_incorrect_pairs = 0.0
        for i in range(n):
            t = targets[i]
            p = predictions[i]
            if t == 1:
                # Number of negative examples that received higher scores than this positive example
                num_incorrect_pairs += ((targets != 1.0) * (predictions >= p)).sum()
            else:
                # Number of positive examples that received lower scores than this negative example
                num_incorrect_pairs += ((targets == 1.0) * (predictions <= p)).sum()
        return num_incorrect_pairs / float(num_pos*num_neg) / 2.0


def print_rank_info(targets, predictions):
    original_targets = targets.copy()

    targets = targets.copy()
    predictions = predictions.copy()

    # Sort the predictions and targets by the prediction value (increasing)
    sorted_indices = np.argsort(predictions)
    predictions = predictions[sorted_indices]
    targets = targets[sorted_indices]

    log.info('Original targets'
    log.info(''.join(['+' if each == 1.0 else '-' for each in original_targets]))
    log.info('Sorted by predictions (%g - %g)' % (predictions[0], predictions[-1]))
    log.info(''.join(['+' if each == 1.0 else '-' for each in targets]))
