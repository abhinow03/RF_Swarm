from setuptools import find_packages, setup

package_name = 'capstone_narrate'

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
    description='RF_Swarm capstone: event-triggered tactical narration '
                '(Layer 2 guard live; Layer 1/3 LLM lazy/dormant until STGT).',
    license='MIT',
    entry_points={
        'console_scripts': [
            'capstone_narrate_node = capstone_narrate.narrate_node:main',
        ],
    },
)
