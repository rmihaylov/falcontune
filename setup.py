from setuptools import setup, find_packages


setup(
    name='falcontune',
    version='0.1.0',
    packages=find_packages(include=['falcontune', 'falcontune.*']),
    entry_points={
        'console_scripts': ['falcontune=falcontune.run:main']
    }
)
