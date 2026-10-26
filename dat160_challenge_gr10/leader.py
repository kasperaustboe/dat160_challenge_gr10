import time
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from dat160_challenge_gr10.prm import PRM
from dat160_challenge_gr10.astar import AStar

class LeaderClass(Node):
    def __init__(self):
        super().__init__('RobotLeaderNode')
        self.prm = None
        
        self.pos_tb3_0 = None
        self.pos_tb3_1 = None
        self.path_sent_tb3_0 = False
        self.path_sent_tb3_1 = False
        
        # Mål for begge roboter (hver sin side)
        self.goal_tb3_0 = (2.0, -3.0) #Høyre side / høyre robot
        self.goal_tb3_1 = (-2.0, 3.0) #Venstre side/venstre robot

        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.clbk_map, map_qos)
        
        self.odom_sub_0 = self.create_subscription(Odometry, '/tb3_0/odom', self.clbk_odom_0, 10)
        self.odom_sub_1 = self.create_subscription(Odometry, '/tb3_1/odom', self.clbk_odom_1, 10)
        
        self.path_pub_0 = self.create_publisher(Path, '/tb3_0/path', 10)
        self.path_pub_1 = self.create_publisher(Path, '/tb3_1/path', 10)
        self.prm_marker_pub = self.create_publisher(MarkerArray, '/prm_graph', 1)

        self.timer = self.create_timer(1.0, self.timer_callback)

    def clbk_odom_0(self, msg):
        self.pos_tb3_0 = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def clbk_odom_1(self, msg):
        self.pos_tb3_1 = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def timer_callback(self):
        if self.prm is not None:
            if self.pos_tb3_0 and not self.path_sent_tb3_0:
                self.send_path(self.pos_tb3_0, self.goal_tb3_0, self.path_pub_0, "tb3_0")
                self.path_sent_tb3_0 = True
                
            if self.pos_tb3_1 and not self.path_sent_tb3_1:
                self.send_path(self.pos_tb3_1, self.goal_tb3_1, self.path_pub_1, "tb3_1")
                self.path_sent_tb3_1 = True

    def send_path(self, start_pos, goal_pos, publisher, name):
        start_node = self.prm.nearest_node(start_pos[0], start_pos[1], visible_only=False)
        goal_node = self.prm.nearest_node(goal_pos[0], goal_pos[1], visible_only=False)

        if start_node is not None and goal_node is not None:
            path_points = AStar.find_path(self.prm, start_node, goal_node)
            if path_points:
                path_msg = Path()
                path_msg.header.frame_id = 'map'
                for (x, y) in path_points:
                    pose = PoseStamped()
                    pose.pose.position.x = float(x)
                    pose.pose.position.y = float(y)
                    path_msg.poses.append(pose)
                publisher.publish(path_msg)

    def clbk_map(self, msg):
        if self.prm is not None: return
        try:
            self.prm = PRM.from_occupancy_grid(msg, cell_size=1.5, points_per_cell=15, neighbors=12, max_edge_length=3.0, robot_radius=0.22)
            if hasattr(self.prm, 'free'):
                from scipy.ndimage import distance_transform_edt
                padded = __import__('numpy').pad(self.prm.free, 1, constant_values=False)
                self.prm.clearance_grid = distance_transform_edt(padded)[1:-1, 1:-1] * self.prm.resolution
            self.publish_prm(msg.header.frame_id or 'map')
        except Exception as e:
            pass

    def publish_prm(self, frame_id='map'):
        if self.prm is None: return
        markers = MarkerArray()
        nodes = Marker()
        nodes.header.frame_id = frame_id
        nodes.ns = 'prm_nodes'
        nodes.id = 1
        nodes.type = Marker.SPHERE_LIST
        nodes.action = Marker.ADD
        nodes.scale.x = nodes.scale.y = nodes.scale.z = 0.08
        nodes.color.r = 1.0; nodes.color.a = 1.0
        for x, y in self.prm.nodes:
            nodes.points.append(Point(x=float(x), y=float(y), z=0.08))
        markers.markers.append(nodes)
        self.prm_marker_pub.publish(markers)

def main(args=None):
    rclpy.init(args=args)
    node = LeaderClass()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()