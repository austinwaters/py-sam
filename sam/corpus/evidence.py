import re


def load_evidence_file(filename, separator="\s+"):
    """
    Reads a data file in 'evidence' format.  Returns a dict of dicts d such that d[instance][feature] contains
    the weight of <feature> in the instance named <instance>.
    """
    result = {}
    for line in open(filename):
        tokens = re.split(separator, line.strip())
        if len(tokens) == 0:
            continue

        name = tokens[0]
        feature_counts = {}
        for s in tokens[1:]:
            feature_id, count = s.split(':')
            feature_counts[feature_id] = float(count)
        result[name] = feature_counts
    return result
