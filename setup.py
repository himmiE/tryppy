from setuptools import setup, find_packages

setup(
    name='tryppy',
    version='0.1.0',
    author='Emely Himmstedt',
    description='Kurze Beschreibung deines Pakets',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/himmiE/tryppy',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    package_data={},
    install_requires=[
        'numpy',
        'scipy',
        'scikit-image',
        'scikit-learn',
        'opencv-python',
        'spatial-efd',
        'tensorflow',
        'keras',
        'matplotlib',
        'shapely',
        'joblib',
        'networkx',
        'pandas',
        'rasterio',
        'tqdm',
        'tifffile',
        'setuptools',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent'
    ],
    extras_require={
        "dev": ["twine>=4.0.2"]
    },
    python_requires='>=3.6',
)