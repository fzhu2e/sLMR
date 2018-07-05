from setuptools import setup, find_packages
import slmr

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='slmr',
    version=slmr.__version__,
    description="A scripting system for the Last Millennium Reanalysis (LMR) project.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Feng Zhu",
    author_email='fengzhu@usc.edu',
    url='https://github.com/fzhu2e/slmr',
    packages=find_packages(),
    include_package_data=True,
    license="MIT license",
    zip_safe=False,
    scripts=['bin/slmr'],
    keywords='slmr',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)
