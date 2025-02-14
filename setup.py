from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()


if __name__=='__main__':
    setup(name='qnmfinder',
          version='0.0.0',
          description='A reverse search algorithm for finding QNMs in the ringdown of NR waveforms.',
          long_description=readme(),
          url='https://github.com/keefemitman/qnmfinder',
          author='Keefe Mitman, Isabella Pretto',
          author_email='kem343@cornelle.edu',
          license='MIT',
          packages=find_packages(),
          install_requires=[
              'matplotlib>=3.9.2',
              'numpy>=1.24.4',
              'numpy_quaternion>=2023.0.4',
              'qnm>=0.4.3',
              'scipy>=1.13.1',
              'scri>=2022.9.0',
              'sxs>=2024.0.27',
              'termcolor>=2.5.0',
              'joblib>=1.4.2'
          ],
          classifiers=[
              "Intended Audience :: Science/Research",
              "License :: OSI Approved :: MIT License",
              "Natural Language :: English",
              "Programming Language :: Python :: 3.9",
              "Topic :: Scientific/Engineering :: Physics",
              "Topic :: Scientific/Engineering :: Astronomy",
              ],
          package_data={'': ['style.mplstyle']},
          include_package_data=True,
          zip_safe=False)
