from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()

setup(
    name='AT_YOLO',
    packages=find_packages(include=['yolo', 'yolo.*']),
    install_requires=parse_requirements('requirements.txt')
)
