import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'delsys_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
         glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dema',
    maintainer_email='5433737@studenti.unige.it',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gui_node = delsys_pkg.gui:main',
            'online_node = delsys_pkg.online_node:main',
            'online_node_emg = delsys_pkg.online_node_emg:main',
            'online_node_raw_imu = delsys_pkg.online_node_raw_imu:main',
            'storing_raw = delsys_pkg.storing_raw:main',
        ],
    },
)
