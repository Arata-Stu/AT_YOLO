from distutils.core import setup

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()

setup(
    name='yolo',
    packages=['AT_YOLO'],
    package_dir={'': 'yolo'},
    install_requires=parse_requirements('requirements.txt')
)
