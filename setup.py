from setuptools import setup

version = '0.10'

with open("README.md", "r") as readme:
    long_description = readme.read()

with open('requirements.txt', 'r') as req:
    install_requires = req.read().splitlines()

project_urls = dict(Source='https://github.com/rileyhales/pyrch',
                    License='https://choosealicense.com/licenses/bsd-3-clause-clear')

classifiers = (
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Natural Language :: English',
)

setup(
    name='rch',
    packages=['rch'],
    version=version,
    install_requires=install_requires,
    python_requires='>3',
    description='A package containing the personal python utilities of Riley Hales',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Riley Hales',
    author_email='rchales@byu.edu',
    url='https://rileyhales.com',
    project_urls=project_urls,
    license='BSD 3-Clause Clear License',
    classifiers=classifiers,
)
