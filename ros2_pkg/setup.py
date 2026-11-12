from setuptools import find_packages, setup

package_name = 'sj_pickplace'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/gazebo_moveit.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='NERO maintainers',
    maintainer_email='maintainer@example.com',
    description='NERO pick & place pipeline (VLM + MCP + MoveIt2)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node  = sj_pickplace.perception_node:main',
            'perception_node_sim = sj_pickplace.perception_node_sim:main',
            'planning_node    = sj_pickplace.planning_node:main',
            'mcp_robot_server = sj_pickplace.mcp_robot_server:main',
            'command_parser   = sj_pickplace.command_parser:main',
            'fake_perception  = sj_pickplace.fake_perception:main',
        ],
    },
)
