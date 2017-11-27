from setuptools import setup

setup(name='cmb_beams',
      version='0.1',
      description='Code for beam convolution.',
      url='https://github.com/oskarkleincentre/cmb_beams',
      author='Adri J. Duivenvoorden',
      author_email='adri.j.duivenvoorden@gmail.com',
      license='MIT',
      packages=['cmb_beams', 'tests'],
#      package_dir={'cmb_beams': 'python'},
      install_requires=['healpy', 'numpy', 'qpoint'],
#      test_suite='nose.collector',
#      test_suite='python/tests.my_test_suite',
      test_suite='tests',
#      tests_require=['nose'],
#      cmdclass={'test' : TestCommand},
      zip_safe=False)
