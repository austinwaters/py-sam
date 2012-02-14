import numpy as np
from math_util import numpy_random_seed_temporarily
from pickle_file_io import PickleFileIO


class DataSet(PickleFileIO):
    def __init__(self, examples, targets, features, keys=None):
        # Make sure examples (and targets) are numpy arrays
        self.examples = np.array(examples)
        self.targets = np.array(targets)
        if not np.isfinite(self.examples).all():
            raise ValueError('Examples may not contain inf of nan entries')
        if not np.isfinite(self.targets).all():
            raise ValueError('Targets may not contain inf of nan entries')

        if keys is not None:
            self.keys = np.array(keys)
        else:
            self.keys = None

        self.features = features
        self.num_instances = self.examples.shape[0]
        self.feature_dim = self.examples.shape[1]  # Note feature_dim may not equal features.get_feature_dim()
        self.classes = np.array(sorted(set(targets)))
        self.num_classes = len(self.classes)
        self.num_per_class = {c:sum(targets == c) for c in self.classes}

        # Compute the scales of the features and targets
        self.feature_scale = np.empty((self.feature_dim, 2))
        self.feature_scale[:, 0] = np.min(self.examples, axis=0)
        self.feature_scale[:, 1] = np.max(self.examples, axis=0)
        self.target_scale = np.array([np.min(self.targets), np.max(self.targets)])

    @classmethod
    def to_signed_labels(cls, zero_one_labels):
        labels_set = set(zero_one_labels)
        if labels_set.difference([0, 1]):
            raise ValueError('Labels contain values other than 0 and 1 (got: %s)' % labels_set)

        result = zero_one_labels.copy()
        result[result == 0] = -1
        return result

    def has_discrete_labels(self):
        target_set = np.array(sorted(set(self.targets)))
        # Detect discrete labels
        return np.all(target_set == np.round(target_set))

    def copy(self):
        return self.subset(np.arange(0, self.num_instances))

    def subset(self, indices):
        if len(indices) == 0:
            raise ValueError('Cannot take empty subset of a DataSet')
        examples = np.take(self.examples, indices, axis=0)
        targets = np.take(self.targets, indices, axis=0)
        if self.keys is not None:
            keys = self.keys[indices]
        else:
            keys = None
        result = DataSet(examples, targets, self.features, keys=keys)
        # Set the feature and target scales of the new dataset to be the same as this dataset.  This information is
        # intended to be used by learners, e.g. in neural network models for initializing the weights and scaling the
        # targets so they fit in the range of the output activation function.  To generalize well, the scale information
        # needs to reflect the whole dataset, rather than just (for example) the subset used for training.
        result.feature_scale = self.feature_scale.copy()
        result.target_scale = self.target_scale.copy()
        return result


    def shuffle(self):
        rands = np.random.rand(self.num_instances)
        shuffled_indices = np.argsort(rands)
        self.examples = self.examples[shuffled_indices, :]
        self.targets = self.targets[shuffled_indices]
        if self.keys:
            self.keys = self.keys[shuffled_indices]

    def weight_by_class(self, weights):
        """
        Generates a new set of examples and targets whose classes are weighted in proportion to the integer
        weights in class_weights.  For class weights {0:1, 1:2}, for instance, this creates a new dataset with two copies
        of each positive example in the original data.
        """
        num_per_class_after_weighting = {c:int(self.num_per_class[c] * weights[c]) for c in self.classes}
        num_total_after_weighting = sum(num_per_class_after_weighting.values())

        weighted_examples = np.empty((num_total_after_weighting, self.feature_dim))
        weighted_targets = np.empty((num_total_after_weighting,))
        if self.keys is not None:
            weighted_keys = np.empty((num_total_after_weighting,), dtype='str')
        else:
            weighted_keys = None

        current = 0
        for i in range(self.num_instances):
            example = self.examples[i, :]
            target = int(self.targets[i])
            key = self.keys[i] if self.keys is not None else None

            num_to_duplicate = weights[target]

            # Duplicate this example 'num_to_duplicate' times in the dataset
            for j in range(num_to_duplicate):
                weighted_examples[current, :] = example
                weighted_targets[current] = target
                if key is not None:
                    weighted_keys[current] = key
                current += 1
        assert current == num_total_after_weighting
        # TODO: copy feature/target scale information
        return DataSet(weighted_examples, weighted_targets, self.features, keys=weighted_keys)

    def __str__(self):
        if self.has_discrete_labels():
            return '<DataSet with dimension %d, %d examples, %d classes, (per class: %s)>' % (self.feature_dim, self.num_instances, self.num_classes, self.num_per_class)
        else:
            return '<DataSet with dimension %d, %d examples>' % (self.feature_dim, self.num_instances)

    def split(self, p, seed=None, stratify=True):
        """
        Performs a stratified split of the dataset.  Returns two DataSets with p and (1-p) fraction of the data,
        respectively.
        """
        if p <= 0.0 or p >= 1.0:
            raise ValueError('Split proportion must be a float between 0 and 1')
        if stratify and not self.has_discrete_labels():
            stratify = False

        with numpy_random_seed_temporarily(seed):
            instances_selected = np.zeros((self.num_instances,), dtype='bool')
            if stratify:
                examples_in_class = {c:np.flatnonzero(self.targets == c) for c in self.classes}
                num_examples_in_class = {c:len(examples_in_class[c]) for c in self.classes}

                for c, indices in examples_in_class.iteritems():
                    np.random.shuffle(indices)

                    # Pick data points for the first split
                    n = int(p * num_examples_in_class[c])
                    if n == 0:
                        raise Exception('Not enough data instances to select %.2f%%' % p*100.)
                    instances_selected[indices[:n]] = True
            else:
                indices = np.arange(self.num_instances)
                np.random.shuffle(indices)
                indices = list(indices)
                
                n = int(p * self.num_instances)
                if n == 0:
                    raise Exception('Not enough data instances to select %.2f%%' % p*100.)
                instances_selected[indices[:n]] = True

        first_indices = np.flatnonzero(instances_selected)
        second_indices = np.flatnonzero(np.negative(instances_selected))
        return self.subset(first_indices), self.subset(second_indices)

    def write_svm_light(self, filename):
        # svm_light requires binary labels
        if self.num_classes > 2:
            raise Exception('Can''t write svm light file for multi-class datasets!')

        with open(filename, 'w') as f:
            # target <feature>:<value> <feature>:<value> ... # comment
            for i in range(self.num_instances):
                target = 1 if self.targets[i] == 1.0 else -1
                features = self.examples[i, :]
                nonzero_feature_indices = np.flatnonzero(features)

                # Feature indices must be >= 1
                features_str = ' '.join(['%d:%g' % (index+1, features[index]) for index in nonzero_feature_indices])
                print >>f, '%d %s  # %s' % (target, features_str, self.keys[i])
