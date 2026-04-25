import os
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

# Check if Cython is available
try:
    from Cython.Build import cythonize
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


class build_ext(_build_ext):
    """Custom build_ext to handle Cython extensions gracefully."""
    
    def run(self):
        if CYTHON_AVAILABLE:
            # Cython is available, build extensions normally
            super().run()
        else:
            # Cython not available, skip building extensions
            # The code will fall back to pure Python implementation
            print("Warning: Cython not available. Building without optimized extensions.")
            print("Install Cython for better performance: pip install cython")


def get_extensions():
    """Get Cython extensions to build."""
    if not CYTHON_AVAILABLE:
        return []
    
    extensions = [
        Extension(
            "plip.structure._pdb_parser",
            ["plip/structure/_pdb_parser.pyx"],
            extra_compile_args=[
                "-O3",              # Maximum optimization
                "-march=native",    # Optimize for current CPU
                "-ffast-math",      # Fast math operations
                "-funroll-loops",   # Unroll loops
            ],
            extra_link_args=["-O3"],
        ),
    ]
    
    # Cython compiler directives
    compiler_directives = {
        'language_level': "3",
        'boundscheck': False,           # Disable bounds checking
        'wraparound': False,            # Disable negative index handling
        'initializedcheck': False,      # Disable initialization checking
        'cdivision': True,              # Use C division (faster)
        'profile': False,               # Disable profiling
        'linetrace': False,             # Disable line tracing
    }
    
    return cythonize(
        extensions,
        compiler_directives=compiler_directives,
        annotate=False,  # Don't generate HTML annotation for release
    )


# Read long description from README
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''


setup(
    name='plip',
    version='3.0.0',
    description='FPLIP - Fast Protein-Ligand Interaction Profiler',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 0 - Alpha',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Cython',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],
    url='https://github.com/BHM-Bob/fplip',
    author='BHM-Bob_G',
    author_email='bhmfly@foxmail.com',
    license='GPLv2',
    packages=find_packages(),
    scripts=['plip/plipcmd.py'],
    cmdclass={'build_ext': build_ext},
    ext_modules=get_extensions(),
    install_requires=[
        'numpy',
        'lxml',
        'openbabel',  # Now available as standard package
    ],
    extras_require={
        'cython': ['cython>=0.29.0'],  # Optional for performance
        'dev': [
            'cython>=0.29.0',
            'pytest',
            'pytest-cov',
        ],
    },
    entry_points={
        "console_scripts": [
            "plip = plip.plipcmd:main"
        ]
    },
    zip_safe=False,
    python_requires='>=3.8',
)