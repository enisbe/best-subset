from setuptools import setup, find_packages

setup(
    name='best-subset',
    version='0.1',
    packages=find_packages(exclude=["sandbox", "sandbox.*"]),
    install_requires=[],
    entry_points={
        'console_scripts': [
            # Add any command line scripts here
        ],
    },
)
