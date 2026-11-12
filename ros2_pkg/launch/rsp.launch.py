import os
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    urdf_path = os.environ.get(
        'RSP_URDF_FILE',
        os.path.join(os.path.dirname(__file__),
                     '..', 'sj_pickplace', 'nero_with_camera.urdf')
    )
    with open(urdf_path) as f:
        urdf = f.read()
    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': urdf}],
        )
    ])
