"""Install Making Language models better few-shot learners."""
import setuptools

def setup_package():
  setuptools.setup(
      name='fewshot',
      version='0.0.1',
      description='PERFECT: Prompt-free and Efficient Few-shot Learning with Language Models',
      author='Rabeeh Karimi Mahabadi',
      license='MIT License',
      packages=setuptools.find_packages(
          exclude=['docs', 'tests', 'scripts', 'examples']),
      install_requires=[
      ],
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',
      ],
      keywords='nlp machinelearning',
  )


if __name__ == '__main__':
  setup_package()
