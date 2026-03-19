from setuptools import setup, find_packages

setup(
    name="bmw-pricing",                 # name of your package
    version="0.1.0",                     # version number
    packages=find_packages(where="src"), # find packages under src/
    package_dir={"": "src"},              # map root package to src directory
)