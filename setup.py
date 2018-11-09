#!/usr/bin/env python

from distutils.core import setup

# Information
__author__ = "Silvia Amabilino"
__copyright__ = "Copyright 2018"
__license__ = "TBC"
__version__ = "1.0.0"
__maintainer__ = "Silvia Amabilino"
__email__ = "silvia.amabilino@bristol.ac.uk"
__status__ = "Beta"
__description__ = "De-novo drug design"
__url__ = "https://github.com/SilviaAmAm/NovaData"
__requirements__ = [
                    "scikit-learn >= 0.19.1",
                    # "tensorflow == 1.9.0",
                    "keras >= 2.2.0"
                    ]

setup(name='NovaData',
      version=__version__,
      description=__description__,
      author=__author__,
      author_email=__email__,
      url=__url__,
      packages=['models'],
      install_requires= __requirements__
     )

