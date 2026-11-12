from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='dat160_challenge_gr10',
            executable='robot_handler',
            output='screen',
        ),
        Node(
            package='dat160_challenge_gr10',
            executable='leader',
            parameters=[{'use_sim_time': True}],
            output='screen',
        ),
    ])