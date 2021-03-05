from setuptools import setup, find_packages


setup(
    name='biospectools',
    version='0.1.0',
    description='Python tools for processing spectral data',
    author='BioSpecNorway Group',
    author_email='?',
    url='https://www.nmbu.no/en/faculty/realtek/research/groups/biospectroscopy',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn'
    ],
    setup_requires=['pytest-runner', 'numpy'],
    tests_require=['pytest']
)
