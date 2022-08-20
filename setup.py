from os.path import abspath, dirname, join
from setuptools import setup

here = abspath(dirname(__file__))

with open(join(here, 'README.md')) as f:
    readme = f.read()

with open(join(here, 'LICENSE')) as f:
    lic = f.read()

setup(name='fhhps',
      version='0.1',
      description='Estimation of random coefficient models',
      author='Jeremy Fox, Vitor Hadad, Stefan Hoderlein, Amil Petrin, Robert Sherman',
      long_description=readme,
      long_description_content_type='text/markdown',
      url='https://github.com/halflearned/FHHPS',
      py_modules=['fhhps'],
      install_requires=[
          "matplotlib>=3.1.1",
          "numpy>=1.17.0",
          "pandas>=0.25.0",
          "scipy>=1.3.0",
          "seaborn>=0.9.0",
          "scikit-learn>=0.20.2"
      ],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3'
      ],
      license=lic)
