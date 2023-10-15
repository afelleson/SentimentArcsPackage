# read version from installed package
# from importlib.metadata import version
# __version__ = version("insert PYPI published package name here")

from .simplifiedSA import * # allows package import via "import imppkg as sa" instead of requiring "import imppkg.simplifiedSA as sa" (the latter still works)