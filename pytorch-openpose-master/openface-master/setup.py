from setuptools import setup

setup(
    name='openface',
    version='0.3.2',
    description="Head pose and fatigue detection based on dlib facial landmarks.",
    url='https://github.com/cmusatyalab/openface',
    packages=['openface'],
    package_data={'openface': ['*.lua']},
)
