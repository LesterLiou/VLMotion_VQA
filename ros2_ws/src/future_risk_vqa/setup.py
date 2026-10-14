from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'future_risk_vqa'


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
    description='Future-oriented person risk pipeline using SAM3, geometry, VLM, and rules.',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'sam3_person_tracker_node = future_risk_vqa.sam3_person_tracker_node:main',
            'ground_trajectory_node = future_risk_vqa.ground_trajectory_node:main',
            'future_vqa_node = future_risk_vqa.future_vqa_node:main',
            'future_policy_node = future_risk_vqa.future_policy_node:main',
            'dual_video_source_node = future_risk_vqa.dual_video_source_node:main',
            'experiment_logger_node = future_risk_vqa.experiment_logger_node:main',
            'research_gui = future_risk_vqa.research_gui:main',
        ],
    },
)
