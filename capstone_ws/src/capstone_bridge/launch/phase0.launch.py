"""Phase 0 bring-up: the 6 stub nodes we own, all with use_sim_time:=true.

capstone_sim is the clock authority (publishes /clock); every other node follows
sim time. predict/narrate are intentionally NOT launched here — they exist
elsewhere and attach during integration.
"""
from launch import LaunchDescription
from launch_ros.actions import Node

SIM_TIME = {'use_sim_time': True}


def generate_launch_description():
    return LaunchDescription([
        Node(package='capstone_sim', executable='capstone_sim_node',
             name='capstone_sim', parameters=[SIM_TIME], output='screen'),
        Node(package='capstone_rff', executable='capstone_rff_node',
             name='capstone_rff', parameters=[SIM_TIME], output='screen'),
        Node(package='capstone_base', executable='capstone_base_node',
             name='capstone_base', parameters=[SIM_TIME], output='screen'),
        Node(package='capstone_track', executable='capstone_track_node',
             name='capstone_track', parameters=[SIM_TIME], output='screen'),
        Node(package='capstone_bridge', executable='capstone_bridge_node',
             name='capstone_bridge', parameters=[SIM_TIME], output='screen'),
        Node(package='capstone_eval', executable='capstone_eval_node',
             name='capstone_eval', parameters=[SIM_TIME], output='screen'),
    ])
