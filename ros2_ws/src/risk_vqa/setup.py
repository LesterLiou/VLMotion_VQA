from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'risk_vqa'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hrc',
    maintainer_email='hrclab.nthu@gmail.com',
    description='ROS 2 visual risk assessment using the VLMotion model worker',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'risk_vqa_node = risk_vqa.risk_vqa_node:main',
            'risk_vqa_gui = risk_vqa.risk_vqa_gui:main',
            'video_ttc_node = risk_vqa.video_ttc_node:main',
            'risk_fusion_node = risk_vqa.risk_fusion_node:main',
            'risk_video_gui = risk_vqa.risk_video_gui:main',
        ],
    },
)
