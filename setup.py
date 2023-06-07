import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = 'rgrav',
    version = '0.0.1',
    author = 'Roman Sermiagin',
    author_email = 'roman.ags@gmail.com',
    description = ('Read CG-6 data file and simple proc of ties'),
    # license = "BSD",
    keywords = 'scintrex cg-6 gravimeter',
    # url = "",
    packages=['cg6_proc'],
    long_description=read('README.md'),
    # classifiers=[
    #     "Development Status :: 1 - Alpha",
    #     "Topic :: Utilities",
    #     "License :: OSI Approved :: BSD License",
    # ],
    install_requires = ['pandas'],
    scripts=['cg6_proc/rgrav.py']
)
