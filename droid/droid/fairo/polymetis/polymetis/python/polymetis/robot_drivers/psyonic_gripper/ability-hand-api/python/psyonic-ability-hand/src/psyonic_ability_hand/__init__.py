__version__ = '0.1.0'

import os

import rich.traceback

os.environ['LOGURU_LEVEL'] = os.getenv('LOGURU_LEVEL', 'INFO')
from loguru import logger as log

rich.traceback.install(show_locals=False)
