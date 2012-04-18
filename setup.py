import setuptools
import setuptools.extension

setuptools.setup(
<<<<<<< HEAD
    name="sam",
    version="0.1",
    packages=setuptools.find_packages(),
    install_requires=[
=======
    name = "py-sam",
    version = "0.1",
    packages = setuptools.find_packages(),
    install_requires = [
>>>>>>> 517b83245b6ca23d1983d533d93c8dc4cbc0f0c5
        #"Cython>=0.15.1",
        "numpy>=1.6.1",
        "scipy>=0.10.0",
        ],
    author = "Austin Waters",
    author_email = "austin@cs.utexas.edu",
    description = "the spherical admixture topic model",
    license = "MIT",
    keywords = "topic models machine learning",
    url = "http://www.cs.utexas.edu/~austin/",
    classifiers = [
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 2.6",
        "Operating System :: Unix",
        "Topic :: Utilities",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )
