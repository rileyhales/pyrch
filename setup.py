from setuptools import setup

with open("README.md", "r") as readme:
    long_description = readme.read()

with open('requirements.txt', 'r') as req:
    install_requires = req.read().splitlines()

setup(
    name='rch',
    packages=['rch'],
    version='0.4',
    description='A package containing the personal python utilities of Riley Hales',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Riley Hales',
    author_email='rchales@byu.edu',
    url='https://rileyhales.com',
    project_urls=dict(Documentation='https://pyrch.readthedocs.io', Source='https://github.com/rileyhales/pyrch',
                      License='https://choosealicense.com/licenses/bsd-3-clause-clear'),
    license='BSD 3-Clause Clear License',
    classifiers=('Programming Language :: Python :: 3',
                 'Topic :: Scientific/Engineering',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Natural Language :: English',),
    install_requires=install_requires
)
