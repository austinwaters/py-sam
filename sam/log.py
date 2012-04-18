from contextlib import contextmanager
import inspect
import logging
import logging.config
import time

# Log levels
NOTSET = logging.NOTSET
DEBUG = logging.DEBUG
WARNING = logging.WARNING
INFO = logging.INFO
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


# Log functions
def debug(*args):
    _logger_for_caller().debug(_join(*args))


def info(*args):
    _logger_for_caller().info(_join(*args))


def warning(*args):
    _logger_for_caller().warning(_join(*args))


def error(*args):
    _logger_for_caller().error(_join(*args))


def critical(*args):
    _logger_for_caller().critical(_join(*args))


def _join(*args):
    'print-style argument joining'
    return ' '.join(map(unicode, args))


def _logger_for_caller():
    "Get the logger for the caller's module"
    caller = inspect.stack()[2][0]
    module = inspect.getmodule(caller)
    if module:
        name = module.__name__
    else:
        name = '__main__'
    return logging.LoggerAdapter(logging.getLogger(name), dict(
        line=caller.f_lineno,  # (The caller's lineno, instead of our own)
    ))
    # TODO Use logging.Filter to transform LogRecords.
    #
    # LogRecord provides various fields, including:
    #
    #   lineno, filename, funcName, module, pathname
    #
    # but our abstraction over the standard logging methods (debug, info, ...)
    # populate these with data from, e.g., our invocation above
    #
    #   _logger_for_caller().debug(...)
    #
    # instead of the client's call
    #
    #   log.debug(...)
    #
    # Using stack inspection (via the inspect module) we could easily recover all
    # the relevant info from the caller, but I got stuck simply when I tried to
    # install a Filter and all the logging messages disappeared. Revisit this
    # sometime.


@contextmanager
def levels(levels):
    'Temporarily set log levels for named loggers'
    oldlevels = {}
    for name, newlevel in levels.items():
        if name == 'root':
            name = ''
        oldlevels[name] = logging.getLogger(name).level
        logging.getLogger(name).setLevel(newlevel)
    yield
    for name, oldlevel in oldlevels.items():
        logging.getLogger(name).setLevel(oldlevel)


class UTCFormatter(logging.Formatter):  # (Used in log.conf)
    'Format time in UTC'
    converter = time.gmtime


# Please forgive me...
_config = {
        'version': 1,
        'formatters': {
            'main': {
                '()': UTCFormatter,
                'format': '[%(asctime)s] %(levelname)s %(message)s (%(name)s:%(line)s)',
                'datefmt': '%Y-%m-%dT%H:%M:%SZ',
                }
            },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'main',
                'stream': 'ext://sys.stdout',
                }
            },
        'root': {
            'level': 'DEBUG',
            'handlers': ['console']
            },
        'loggers': {
            'sam': {
                'level': 'DEBUG'
                }
            }
        }
logging.config.dictConfig(_config)
