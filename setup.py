from distutils.core import setup

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()

setup(
    name='AT_YOLO',
    packages=['yolo'],
    package_dir={'': 'AT_YOLO'},
    install_requires=parse_requirements('requirements.txt')
)
