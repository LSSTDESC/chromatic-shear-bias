import os
from setuptools import setup, find_packages

scripts = [
    "bin/run-chromatic-shear-bias",
    "bin/meas-chromatic-shear-bias",
    "bin/validate-seds",
    "bin/viz-chromatic-shear-bias",
]

setup(
    name="chromatic-shear-bias",
    version=0.1,
    description="Studies of the contribution to shear calibration bias from chromatic effects ",
    author="smau",
    packages=find_packages(),
    scripts=scripts,
)
