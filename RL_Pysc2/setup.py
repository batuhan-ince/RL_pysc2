from setuptools import setup, find_packages

setup(
    # Metadata
    name='RL_pysc2',
    version=0.1,
    author='Tolga Ok,Emircan Kılıçkaya,Batuhan Ince',
    author_email='inceb97@gmail.com',
    url='',
    description='Starcraft II gym environment for reinforcement learning research',
    long_description="",
    license='MIT',

    # Package info
    packages=["rl_pysc2", ],
    install_requires=["gym",
                      "torch",
                      "pysc2",
                      "numpy",
                      ],
    zip_safe=False
)
