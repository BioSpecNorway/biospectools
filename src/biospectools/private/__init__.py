import sys
import logging

try:
    from biospectools_private import preprocessing
    from biospectools_private import physics
    from biospectools_private import data
    from biospectools_private import models

    for module in (preprocessing, physics, data, models):
        full_name = '{}.{}'.format(__package__, module.__name__.rsplit('.')[-1])
        sys.modules[full_name] = sys.modules[module.__name__]
except ImportError:
    logging.exception('biospectools_private is not installed')
