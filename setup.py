from setuptools import find_packages, setup

setup(
    name="my_gym",
    packages=[package for package in find_packages() if package.startswith("my_gym")],
    url="https://github.com/osigaud/my_gym",
    version="0.0.1",
    install_requires=[
        "git+https://github.com/osigaud/SimpleMazeMDP.git",
        "gym==0.21.0",
        "numpy>=1.19.1",
        "Box2D",
    ],
    tests_require=["pytest==4.4.1"],
    test_suite="tests",
    author="Olivier Sigaud",
    author_email="Olivier.Sigaud@isir.upmc.fr",
    license="MIT",
    description="A set of additional gym environments",
    long_description=open("README.md").read(),
)
