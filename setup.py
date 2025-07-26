from setuptools import setup, find_packages

reqs=[
    ]

setup(
    name='CTDGG',
    version='0.0.1',
    url=None,
    author='Antoine Siraudin',
    author_email='antoine.siraudin@student-cs.fr',
    packages=find_packages(exclude=["configs"]),
    install_requires=reqs
)