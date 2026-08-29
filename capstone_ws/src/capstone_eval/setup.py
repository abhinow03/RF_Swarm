from setuptools import find_packages, setup

package_name = 'capstone_eval'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pw26_akp_01',
    maintainer_email='wabhinow223@gmail.com',
    description='RF_Swarm capstone: metrics -> CSV. Only package allowed to read truth (Phase 0 stub).',
    license='MIT',
    entry_points={
        'console_scripts': [
            'capstone_eval_node = capstone_eval.eval_node:main',
        ],
    },
)
