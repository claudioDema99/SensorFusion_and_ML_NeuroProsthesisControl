from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='delsys_pkg',
            executable='online_node',
            name='online_node'
        ),
        TimerAction(
            period=1.0,
            actions=[Node(
                package='delsys_pkg',
                executable='online_node_emg',
                name='online_node_emg'
            )]
        ),
        TimerAction(
            period=2.0,
            actions=[Node(
                package='delsys_pkg',
                executable='online_node_raw_imu',
                name='online_node_raw_imu'
            )]
        ),
        TimerAction(
            period=3.0,
            actions=[Node(
                package='delsys_pkg',
                executable='gui_node',
                name='gui_node'
            )]
        )
    ])