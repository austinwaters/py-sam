import setuptools
import setuptools.extension

setuptools.setup(
    name = "sam",
    version = "0.1",
    packages = setuptools.find_packages(),
    install_requires = [
        #"Cython>=0.15.1",
        "numpy>=1.6.1",
        "scipy>=0.10.0",
        ],
    )

