from setuptools import setup, find_packages

setup(
    name='tony_ml',
    version='1.0.4',
    packages=find_packages(exclude=['main']),
    install_requires=[],
    url='https://github.com/anthonygfruit/AutoML',
    license='Apache 2.0',
    author='AFruit',
    author_email='anthony.fruit@gmail.com',
    description='Testing out some automl shizzle'
)