from ast import literal_eval
from distutils.core import setup


def get_version(source='vcfplt.py'):
    with open(source) as f:
        for line in f:
            if line.startswith('__version__'):
                return literal_eval(line.partition('=')[2].lstrip())
    raise ValueError("__version__ not found")


setup(
    name='vcfplt',
    version=get_version(),
    author='Alistair Miles',
    author_email='alimanfoo@googlemail.com',
    py_modules=['vcfplt'],
    url='https://github.com/alimanfoo/vcfplt',
    license='MIT License',
    description='Convenient plotting functions for variant call data.',
    long_description=open('README.md').read(),
    classifiers=['Intended Audience :: Developers',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python',
                 'Topic :: Software Development :: Libraries :: Python Modules'
                 ]
)
