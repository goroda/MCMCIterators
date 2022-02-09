"""Setup for MCMCIterators

setup.py heavily borrowed from 
https://github.com/pypa/sampleproject

See: 
https://packaging.python.org/guides/distributing-packages-using-setuptools/

"""
import pathlib
from setuptools import setup, find_packages


here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Arguments marked as "Required" below must be included for upload to pyPI
# Fields markedas optional may be commented out

setup(

    name='MCMCIterators',
    version='0.0.0',
    description='Markov Chain Monte Carlo Iterators',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=None,
    author='Alex Gorodetsky',
    author_email='alex@alexgorodetsky.com',
    # classifiers=[
    # ]
    # keywords='sample,setuptools'
    packages=['MCMCIterators']
)
