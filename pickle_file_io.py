import cPickle as pickle
import os


class PickleFileIO(object):
    """
    Mixin that provides convenience functions for doing IO with pickle files.
    """

    def save(self, path):
        """
        Saves an object to file.  The file is written completely to a temporary location and, upon completion, renamed
        to 'path'.  This is intended to minimize the risk of leaving the file in a partially written state, e.g.
        when the disk is full or your job is pre-empted in the middle of the write (it happens).
        """
        if not os.path.isfile(path):
            pickle.dump(self, open(path, 'wb'))
        else:
            # Write the object to a temporary file, then copy at the last minute
            temp_path = path + '.tmp'
            try:
                pickle.dump(self, open(temp_path, 'wb'))
                os.rename(temp_path, path)
            finally:
                if os.path.isfile(temp_path):
                    os.remove(temp_path)

    @classmethod
    def load(cls, path, check_class=True):
        """
        Loads an object from file, checking that the instance matches the type of the receiver class.
        """
        if not os.path.exists(path):
            raise Exception("Cannot find %s!" % path)
        instance = pickle.load(open(path))

        # Check that we loaded the correct type of object
        if not isinstance(instance, cls) and check_class:
            class_name = instance.__class__.__name__
            full_class_name = instance.__class__.__module__ + '.' + class_name

            expected_full_class_name = cls.__module__ + '.' + cls.__name__
            raise Exception("Loaded object of type %s (%s) from file %s (expected %s or subclass)" % (class_name, full_class_name, path, expected_full_class_name))
        instance.post_load()
        return instance

    def post_load(self):
        pass
