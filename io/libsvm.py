import re


def load_libsvm(filename):
    """
    Reads a data file in libsvm format.  Returns a dict of dicts d such that d[instance][feature] contains
    the count of <feature> in the instance named <instance>.
    """
    result = {}
    for line in open(filename):
        tokens = re.split('\s+', line.strip())
        if len(tokens) == 0:
            continue

        name = tokens[0]
        feature_counts = {}
        for s in tokens[1:]:
            feature_id, count = s.split(':')
            feature_counts[feature_id] = int(count)
        result[name] = feature_counts
    return result