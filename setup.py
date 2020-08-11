import setuptools
from setuptools import Command
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    'matplotlib',
    'numpy',
    'scipy'
]

class CleanCommand(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')

cmdclass = {'clean': CleanCommand}

setuptools.setup(
    name="ConstrainedKMeans",
    package_dir={'ConstrainedKMeans': 'src'},
    packages=['ConstrainedKMeans'],
    version="1.2",
    author="Euxhen Hasanaj",
    author_email="ehasanaj@cs.cmu.edu",
    description="A constrained KMeans algorithm.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ferrocactus/ConstrainedKMeans",
    install_requires=install_requires,
    cmdclass=cmdclass,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
