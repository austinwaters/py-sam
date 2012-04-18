import numpy as np


class ArffWriter(object):
    def __init__(self, filename, dim=None, class_list=None, relation_name='relation'):
        # TODO: support non-categorical targets
        assert class_list is not None
        self.filename = filename
        self.relation_name = relation_name
        self.dim = dim
        self.class_list = [str(each) for each in class_list]

        self.f = open(self.filename, 'w')
        self.header_written = False

    def write_header(self):
        print >>self.f, '@RELATION %s' % self.relation_name
        for i in range(self.dim):
            print >>self.f, '@ATTRIBUTE feature%d NUMERIC' % i
        # XXX: we're assuming targets will always be categorical class labels
        print >>self.f, '@ATTRIBUTE class {%s}' % ','.join(self.class_list)
        print >>self.f, '@DATA'
        self.header_written = True

    def write_example(self, example, target):
        if self.f is None:
            raise Exception('Cannot write example on closed ArffWriter')

        example = np.asarray(example)
        target = str(target)
        if example.ndim != 1:
            raise ValueError('Example must be a 1-D numeric vector')

        if self.dim is None:
            self.dim = len(example)
        else:
            if len(example) != self.dim:
                raise ValueError('Example has incorrect dimension (expected %d; got %d)' % (len(example), self.dim))

        if not self.header_written:
            self.write_header()

        features_string = ', '.join(['%g' % each for each in example])
        print >>self.f, '%s, %s' % (features_string, target)

    def close(self):
        if self.f is None:
            return
        self.f.close()
        self.f = None
