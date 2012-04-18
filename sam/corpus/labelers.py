"""
Defines labelers for evidence files, i.e. functions that map the document names to their class labels.
"""

import os


def twenty_news(name):
    # E.g. 20news-bydate-train/alt.atheism/51154 => alt.athiesm
    parts = name.split('/')
    if len(parts) != 3:
        raise ValueError('Error parsing class label out of 20news id "%s"' % name)
    return parts[1]


def dirname(name):
    # Ex. /foo/bar/bedroom/image_0001.jpg.png => bedroom
    return os.path.basename(os.path.dirname(os.path.abspath(name)))


# Register labelers here so they can be referenced by name on the command line
registry = {
    '20news':twenty_news,
    '13scene':dirname,
    'dirname':dirname,
}
