from setuptools import setup

version = '0.15'

with open("README.md", "r") as readme:
    long_description = readme.read()

PYTHON_REQUIRES = '>=3.11'
INSTALL_REQUIRES = [
    'google-api-python-client',
    'google-auth-httplib2',
    'google-auth-oauthlib',
    'affine',
    'numpy',
    'pandas',
    'geopandas',
    'requests',
    'shapely',
    'rasterio'
]

PROJECT_URLS = dict(Source='https://github.com/rileyhales/pyrch',
                    License='https://choosealicense.com/licenses/bsd-3-clause-clear')

classifiers = [
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Natural Language :: English',
]

setup(
    name='rch',
    packages=['rch'],
    version=version,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    description='A package containing the personal python utilities of Riley Hales',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Riley Hales',
    author_email='rchales@byu.edu',
    url='https://rileyhales.com',
    project_urls=PROJECT_URLS,
    license='BSD 3-Clause Clear License',
    classifiers=classifiers,
)
