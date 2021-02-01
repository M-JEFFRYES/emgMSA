from setuptools import setup, Extension

with open('README.rst') as f:
    readme = f.read()

setup(
    name='muscle-synergy-analysis',
    author='Michael Jeffryes',
    author_email='mike.jeffryes@hotmail.com',
    url='',
    version='1.0.0',
    description='Analyses emg muscle synergies ',
    #packages=['emgmsa'],
    py_modules=["MSA",],
    package_dir={'':'src'},
    setup_requires=['wheel'],
    classifiers=[
        #"License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    #install_requires=[],
    long_description=readme,
    long_description_content_type='text/markdown',
    #test_suite='setup.my_test_suite',
)

# python setupy.py sdist bdist_wheel
# twine upload dist/* --skip-existing