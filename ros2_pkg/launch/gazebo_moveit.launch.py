"""
gazebo_moveit.launch.py
========================
nero 로봇팔 Gazebo + MoveIt2 통합 launch
- Gazebo 11 (classic) 시뮬
- ros2_control (gazebo_ros2_control plugin)
- MoveIt2 move_group
- RViz2

실행:
  ros2 launch sj_pickplace gazebo_moveit.launch.py
"""

import os
import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
    RegisterEventHandler, TimerAction
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():

    # ── 경로 ──────────────────────────────────
    pkg_moveit = get_package_share_directory('nero_gripper_moveit_config')
    pkg_desc   = get_package_share_directory('agx_arm_description')
    gazebo_ros = get_package_share_directory('gazebo_ros')

    # ── URDF: nero_gripper_d435.urdf에 gazebo plugin 주입 ──
    urdf_path = os.path.join(
        pkg_desc, '..', '..', '..', '..',
        'src', 'agx_arm_sim', 'agx_arm_description', 'urdf', 'nero_gripper_d435.urdf'
    )
    urdf_path = os.path.realpath(urdf_path)

    with open(urdf_path, 'r') as f:
        urdf_raw = f.read()

    # gazebo_ros2_control plugin 삽입 (</robot> 바로 앞)
    gazebo_plugin = """
  <gazebo>
    <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
      <robot_sim_type>gazebo_ros2_control/GazeboSystem</robot_sim_type>
      <parameters>{ros2_controllers_yaml}</parameters>
    </plugin>
  </gazebo>
""".format(ros2_controllers_yaml=os.path.join(pkg_moveit, 'config', 'ros2_controllers.yaml'))

    robot_description_content = urdf_raw.replace('</robot>', gazebo_plugin + '</robot>')

    robot_description = {'robot_description': robot_description_content}

    # ── MoveIt2 설정 ──────────────────────────
    moveit_config = (
        MoveItConfigsBuilder('nero', package_name='nero_gripper_moveit_config')
        .robot_description(file_path=urdf_path)
        .to_moveit_configs()
    )

    # ── 노드들 ────────────────────────────────

    # 1. robot_state_publisher
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description],
    )

    # 2. Gazebo 실행
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'verbose': 'false', 'pause': 'false'}.items(),
    )

    # 3. 로봇 스폰
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'nero_arm',
            '-x', '0', '-y', '0', '-z', '0.01',
        ],
        output='screen',
    )

    # 4. joint_state_broadcaster
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    # 5. arm_controller (스폰 후 실행)
    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    # 6. gripper_controller
    gripper_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    # 7. move_group
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            moveit_config.to_dict(),
            {'use_sim_time': True},
        ],
    )

    # 8. RViz2
    rviz_config = os.path.join(pkg_moveit, 'config', 'moveit.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            {'use_sim_time': True},
        ],
    )

    # ── 순서 제어: 스폰 완료 후 컨트롤러 spawn ──
    spawn_done_jsb = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_robot,
            on_exit=[joint_state_broadcaster_spawner],
        )
    )
    jsb_done_arm = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[arm_controller_spawner, gripper_controller_spawner],
        )
    )
    # move_group은 컨트롤러 뜬 후 3초 뒤
    arm_done_moveit = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=arm_controller_spawner,
            on_exit=[
                TimerAction(period=3.0, actions=[move_group_node, rviz_node])
            ],
        )
    )

    return LaunchDescription([
        rsp_node,
        gazebo,
        TimerAction(period=2.0, actions=[spawn_robot]),
        spawn_done_jsb,
        jsb_done_arm,
        arm_done_moveit,
    ])
