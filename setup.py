from setuptools import setup, find_packages

setup(
    name = 'darknet_detector',
    packages = find_packages(),
    scripts = ['darknet_detector/vid_streamer.py'],
    install_requires = [
        'scipy',
        'filterpy==1.4.1', 
        'numba==0.38.1', 
        'scikit-image==0.15.0', 
        'scikit-learn==0.19.1',
        'opencv-python'
     ],
    dependency_links = [
        'git+https://github.com/aleju/imgaug'
    ],
    package_data = {
        'darknet_detector' : [
            '*.dll', '*.weights',
            'cfg/*.cfg', 'cfg/*.data', 
            'data/*.names'
        ]
    },
    url = "https://github.com/joei1994/darknet_detector/tree/master"
)
