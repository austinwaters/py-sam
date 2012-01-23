"""
Support for running Python jobs on Condor.

Subclasses of Condorizable should implement two methods, check_args(argv) and run(options).

check_args(argv): parses a list of command-line arguments and, if possible, checks whether the job will be able to be
run successfully on Condor.  The job might not be able to run, for instance, because an input file listed in the
arguments doesn't exist.  This function is run locally, before the job is shipped off to Condor, so this allows jobs to
"fail fast".  Returns an options namespace (see argparse.ArgumentParser).

run(options): given an options namespace, runs the task.
"""
# TODO: add doc for locked output files


import inspect
import os
import signal
import sys
from tempfile import mkstemp

CondorScriptHeader = """
universe = vanilla
requirements = %s

Initialdir = %s
Executable = %s
getenv = True
kill_sig = 15
"""

CondorScriptFooter = """
+Group   = "GRAD"
+Project = "AI_ROBOTICS"
+ProjectDescription = "Woo!"

Arguments = %s
Queue
"""

LogOutputPart = """
Output = %s
stream_output = True
"""

LogErrorPart = """
Error =  %s
stream_error = True
"""

def kwargs_to_argv(kw):
    argv = []
    for k, v in kw.items():
        if v == '':
            argv.append('--%s' % k)
        elif type(v) == str:
            argv.append("--%s='%s'" % (k, v))
        else:
            argv.append('--%s=%s' % (k, v))
    return argv


class Condorizable(object):
    CONDOR_FLAG = '--condor'
    CONDOR_LOG_FLAG = '--log'
    PYTHON = sys.executable
    binary = None

    def __init__(self, argv=None, kw=None):
        self.output_files = []

        if kw and argv:
            raise ValueError('Must provide at most one of the argv or kw kwargs')

        # If the class doesn't specify 'binary', use the python file in which the class was defined.
        if self.binary is None:
            self.binary = os.path.abspath(inspect.getfile(self.__class__))
            print 'Guessing binary %s' % self.binary
        
        if not os.path.isfile(self.binary):
            raise Exception('Unable to locate binary %s for condorizable job' % self.binary)
            
        if kw is not None:
            argv = [self.binary] + kwargs_to_argv(kw)
        elif argv is not None:
            argv = list(argv)

        # Install the sigterm handler
        signal.signal(signal.SIGTERM, self.sigterm_handler)

        self.argv = argv
        if self.argv is not None:
            self.parse_argv_and_run(self.argv)

    @classmethod
    def path_to_script(cls, filename):
        if filename.endswith('.pyc') or filename.endswith('.pyo'):
            filename = os.path.splitext(filename)[0] + '.py'
        return os.path.abspath(filename)

    def error(self, message):
        raise RuntimeError(message)

    def sigterm_handler(self, signum=None, frame=None):
        self.on_exit()
        sys.exit(1)

    def on_exit(self):
        for output_file in self.output_files:
            lock_file = self.get_lock_file_for(output_file)
            if os.path.isfile(lock_file):
                print 'Removing lock file %s' % lock_file
                os.remove(lock_file)

    def add_output_file(self, filename):
        self.output_files.append(filename)

    def lock_output_files_or_die(self):
        for each in self.output_files:
            if not self.lock_output_file(each):
                raise Exception('Unable to lock output file %s; another process may be writing to this file' % each)

    def lock_output_file(self, filename):
        lock_filename = self.get_lock_file_for(os.path.abspath(filename))
        try:
            # This call should be atomic at the OS level, but it (probably) won't protect against multiple machines
            # trying to create the same lock file on NFS.  This fails with an OSError if the lock file already exists.
            f = os.fdopen(os.open(lock_filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL), 'w')
            f.close()
            return True
        except OSError, e:
            return False

    @classmethod
    def get_lock_file_for(cls, filename):
        filename = os.path.abspath(filename)
        return os.path.join(os.path.dirname(filename), '.' + os.path.basename(filename) + '.lock')

    @classmethod
    def is_locked(cls, filename):
        lock_file = cls.get_lock_file_for(filename)
        return os.path.isfile(lock_file)

    @classmethod
    def check_output_file_is_unlocked(cls, filename):
        if cls.is_locked(filename):
            raise Exception(
                'Found lock for output file %s!  Another process may be writing to this file.' % filename)

    def parse_argv_and_run(self, argv=None):
        if argv is not None:
            self.argv = argv

        condorize = self.CONDOR_FLAG in self.argv
        if condorize:
            self.argv.remove(self.CONDOR_FLAG)
        log_output = self.CONDOR_LOG_FLAG in self.argv
        if log_output:
            self.argv.remove(self.CONDOR_LOG_FLAG)
            if not condorize:
                raise Exception('Flag %s only applies to condor jobs' % self.CONDOR_LOG_FLAG)

        # Check the arguments, even if this is a condor job being started.  This allows condor jobs to fail fast,
        # rather than dying on a remote node.
        options = self.check_args(self.argv)
        if options is None:
            raise Exception('check_args function must return options structure; got None instead')

        for filename in self.output_files:
            self.check_output_file_is_unlocked(filename)

        if condorize:
            print 'Condorizing %s' % ' '.join(self.argv)
            self.run_on_condor(self.argv, log_output=log_output)
            return

        try:
            self.lock_output_files_or_die()
            self.run(options)
        finally:
            self.on_exit()

    def run_on_condor(self, argv, requirements=None, log_output=False):
        executable = self.PYTHON
        args = '-O ' + ' '.join(argv)
        current_dir = os.getcwd()
        requirements = 'InMastodon && (%s)' % requirements if requirements else 'InMastodon' 

        temp_prefix = '%s.%s-' % (os.path.basename(executable), str(os.getpid()))
        condor_file = mkstemp(dir='/tmp', prefix=temp_prefix, suffix='.condor')[1]
        with open(condor_file, 'w') as f:
            f.write(CondorScriptHeader % (requirements, current_dir, executable))

            if log_output:
                log_filename = os.path.join(current_dir, os.path.basename(condor_file) + '.out')
                f.write(LogOutputPart % log_filename)
                log_filename = os.path.join(current_dir, os.path.basename(condor_file) + '.err')
                f.write(LogErrorPart % log_filename)

            f.write(CondorScriptFooter % args)
        # TODO check if this fails
        os.popen('condor_submit %s' % condor_file)


    def check_args(self, argv):
        """ Parses the argument list and returns an options structure. """
        raise NotImplementedError

    def run(self, options):
        raise NotImplementedError
