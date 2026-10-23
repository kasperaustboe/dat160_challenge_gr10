import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from dat160_challenge_gr10.prm import PRM


class LeaderClass(Node):
    def __init__(self):
        super().__init__('RobotLeaderNode')

        self.prm = None

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.clbk_map,
            map_qos
        )

        self.prm_marker_pub = self.create_publisher(
            MarkerArray,
            '/prm_graph',
            1
        )

        # TODO: Abonnementer for robot_handler.
        self.tb3_0_lidar_value = 1000
        self.tb3_1_lidar_value = 1000

        timer_period = 1.0

        self.timer = self.create_timer(
            timer_period,
            self.timer_callback
        )

    def clbk_tb3_0_test(self, msg):
        self.tb3_0_lidar_value = msg.data

    def clbk_tb3_1_test(self, msg):
        self.tb3_1_lidar_value = msg.data

    def timer_callback(self):
        self.get_logger().info(
            'TB3_0 Lidar Value: ' + str(self.tb3_0_lidar_value)
        )

        self.get_logger().info(
            'TB3_1 Lidar Value: ' + str(self.tb3_1_lidar_value)
        )

    def clbk_map(self, msg):
        if self.prm is not None:
            return

        self.get_logger().info(
            f'Mottok kart: {msg.info.width}x{msg.info.height}'
        )

        start_time = time.perf_counter()

        try:
            self.prm = PRM.from_occupancy_grid(
                msg,
                cell_size=2.0,
                points_per_cell=12, #Endre antall punkter 
                neighbors=12,
                max_edge_length=4.0,
                robot_radius=0.22
            )

            elapsed = time.perf_counter() - start_time

            edge_count = (
                sum(len(connections) for connections in self.prm.edges.values())
                // 2
            )

            self.get_logger().info(
                f'PRM generert: {len(self.prm.nodes)} punkter, '
                f'{edge_count} forbindelser. '
                f'Tid: {elapsed:.2f} sekunder'
            )

            self.publish_prm(
                msg.header.frame_id or 'map'
            )

        except Exception as e:
            self.prm = None

            self.get_logger().error(
                f'Feil ved generering av PRM: {e}'
            )

    def publish_prm(self, frame_id='map'):
        if self.prm is None:
            return

        markers = MarkerArray()

        # Publiser forbindelsene mellom punktene til prm
        edges = Marker()

        edges.header.frame_id = frame_id
        edges.header.stamp = self.get_clock().now().to_msg()
        edges.ns = 'prm_edges'
        edges.id = 0
        edges.type = Marker.LINE_LIST
        edges.action = Marker.ADD

        edges.scale.x = 0.025

        edges.color.r = 0.1
        edges.color.g = 0.8
        edges.color.b = 1.0
        edges.color.a = 0.8

        for node_id, connections in self.prm.edges.items():
            x1, y1 = self.prm.nodes[node_id]

            for other_id in connections:
                if node_id >= other_id:
                    continue

                x2, y2 = self.prm.nodes[other_id]

                edges.points.append(
                    Point(x=float(x1), y=float(y1), z=0.05)
                )

                edges.points.append(
                    Point(x=float(x2), y=float(y2), z=0.05)
                )

        markers.markers.append(edges)

        # Publiser selve PRM-punktene.
        nodes = Marker()

        nodes.header.frame_id = frame_id
        nodes.header.stamp = self.get_clock().now().to_msg()
        nodes.ns = 'prm_nodes'
        nodes.id = 1
        nodes.type = Marker.SPHERE_LIST
        nodes.action = Marker.ADD

        nodes.scale.x = 0.08
        nodes.scale.y = 0.08
        nodes.scale.z = 0.08

        nodes.color.r = 1.0
        nodes.color.g = 0.2
        nodes.color.b = 0.1
        nodes.color.a = 1.0

        for x, y in self.prm.nodes:
            nodes.points.append(
                Point(x=float(x), y=float(y), z=0.08)
            )

        markers.markers.append(nodes)

        self.prm_marker_pub.publish(markers)

        self.get_logger().info(
            'Publiserte PRM-graf til /prm_graph'
        )


def main(args=None):
    rclpy.init(args=args)

    robot_leader = LeaderClass()

    try:
        rclpy.spin(robot_leader)
    except KeyboardInterrupt:
        pass
    finally:
        robot_leader.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()