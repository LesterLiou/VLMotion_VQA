from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'risk_vqa_sim'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'docs'), glob('docs/*.md')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hrc',
    maintainer_email='hrclab.nthu@gmail.com',
    description='Dual-camera temporal VQA and deterministic risk decisions for Isaac Sim.',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'dual_camera_vqa_node = risk_vqa_sim.dual_camera_vqa_node:main',
            'dual_video_source_node = risk_vqa_sim.dual_video_source_node:main',
            'risk_vqa_sim_gui = risk_vqa_sim.risk_vqa_sim_gui:main',
            'risk_decision_node = risk_vqa_sim.risk_decision_node:main',
        ],
    },
)
