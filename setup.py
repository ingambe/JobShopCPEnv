from setuptools import setup

from mypyc.build import mypycify

setup(name='jss_approach',
      version='1.0.0',
      author="Pierre Tassel",
      author_email="pierre.tassel@aau.at",
      description="An optimized RL approach to learn and simulate"
                  " the Job-Shop Scheduling problem using Constraint Programming.",
      url="https://github.com/ingambe",
      packages=['compiled_jss'],
      package_dir={'': 'src'},
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      python_requires='>=3.8',
      install_requires=['mypy', 'gym', 'numpy', 'matplotlib', 'docplex', 'cloudpickle'],
      tests_require=['pytest', 'hypothesis'],
      include_package_data=True,
      ext_modules=mypycify([
          'src/compiled_jss/CPEnv.py',
      ]),
)
