from distutils.core import setup

script_files = ['bin/simvideo']

setup(
    name='SimVideo',
    version='2.2',
    author='Wolfgang Kastaun',
    author_email='physik@fangwolg.de',
    packages=['simvideo', 'simvideo.video'],
    scripts=script_files,
    description='Infrastructure for making movies from Cactus data.',
    long_description=open('README.rst').read(),
    install_requires=[
        "PostCactus", "h5py",
        "numpy", "scipy", "matplotlib",
        'futures; python_version == "2.7"'
    ],
    python_requires='>=2.7',
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ]
)

