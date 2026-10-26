import math
import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from dat160_challenge_gr10.prm import PRM
from dat160_challenge_gr10.astar import AStar


class ExplorationManager:
    def __init__(self, map_width, map_height, resolution, origin_x, origin_y, center_x):
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.center_x = center_x
        
        self.cols = int(map_width / resolution)
        self.rows = int(map_height / resolution)
        self.visit_grid = np.zeros((self.rows, self.cols), dtype=int)

    def _pos_to_grid(self, x, y):
        adj_x = x - self.origin_x
        adj_y = y - self.origin_y
        col = int(adj_x / self.resolution)
        row = int(adj_y / self.resolution)
        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))
        return row, col

    def update_robot_position(self, x, y):
        row, col = self._pos_to_grid(x, y)
        self.visit_grid[row][col] += 1

    def get_next_target(self, robot_id, prm_nodes, current_x, current_y):
        valid_nodes = []
        for node_x, node_y in prm_nodes:
            if robot_id == "tb3_0" and node_x < self.center_x:
                valid_nodes.append((node_x, node_y))
            elif robot_id == "tb3_1" and node_x >= self.center_x:
                valid_nodes.append((node_x, node_y))

        if not valid_nodes:
            return None
            
        node_scores = []
        for x, y in valid_nodes:
            row, col = self._pos_to_grid(x, y)
            visits = self.visit_grid[row][col]
            node_scores.append((visits, x, y))
            
        min_visits = min(score[0] for score in node_scores)
        best_candidates = [(nx, ny) for visits, nx, ny in node_scores if visits == min_visits]
        
        best_node = None
        min_dist = float('inf')
        for nx, ny in best_candidates:
            dist = math.dist((current_x, current_y), (nx, ny))
            if dist < min_dist:
                min_dist = dist
                best_node = (nx, ny)
                
        return best_node


class LeaderClass(Node):
    def __init__(self):
        super().__init__('RobotLeaderNode')
        self.prm = None
        self.manager = None
        
        self.pos_tb3_0 = None
        self.pos_tb3_1 = None
        
        # Flagg for å styre når robotene skal få ny rute
        self.is_exploring_tb3_0 = False
        self.is_exploring_tb3_1 = False

        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.clbk_map, map_qos)
        
        self.odom_sub_0 = self.create_subscription(Odometry, '/tb3_0/odom', self.clbk_odom_0, 10)
        self.odom_sub_1 = self.create_subscription(Odometry, '/tb3_1/odom', self.clbk_odom_1, 10)
        
        # Lytter til status fra RobotHandler for å vite når de er ferdige med ruten/skanningen
        self.status_sub_0 = self.create_subscription(String, '/tb3_0/status', self.clbk_status_0, 10)
        self.status_sub_1 = self.create_subscription(String, '/tb3_1/status', self.clbk_status_1, 10)

        self.path_pub_0 = self.create_publisher(Path, '/tb3_0/path', 10)
        self.path_pub_1 = self.create_publisher(Path, '/tb3_1/path', 10)
        self.prm_marker_pub = self.create_publisher(MarkerArray, '/prm_graph', 1)

        self.timer = self.create_timer(1.0, self.timer_callback)

    def clbk_odom_0(self, msg):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        self.pos_tb3_0 = (x, y)
        if self.manager:
            self.manager.update_robot_position(x, y)

    def clbk_odom_1(self, msg):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        self.pos_tb3_1 = (x, y)
        if self.manager:
            self.manager.update_robot_position(x, y)

    def clbk_status_0(self, msg):
        if msg.data == "READY":
            self.is_exploring_tb3_0 = False

    def clbk_status_1(self, msg):
        if msg.data == "READY":
            self.is_exploring_tb3_1 = False

    def timer_callback(self):
        # Tildeler mål bare hvis kartet er klart og roboten for øyeblikket står stille (ikke utforsker)
        if self.prm is not None and self.manager is not None:
            if self.pos_tb3_0 and not self.is_exploring_tb3_0:
                self.assign_new_target("tb3_0", self.pos_tb3_0, self.path_pub_0)
                
            if self.pos_tb3_1 and not self.is_exploring_tb3_1:
                self.assign_new_target("tb3_1", self.pos_tb3_1, self.path_pub_1)

    def assign_new_target(self, robot_id, start_pos, publisher):
        target = self.manager.get_next_target(robot_id, self.prm.nodes, start_pos[0], start_pos[1])
        if target:
            success = self.send_path(start_pos, target, publisher)
            if success:
                if robot_id == "tb3_0":
                    self.is_exploring_tb3_0 = True
                else:
                    self.is_exploring_tb3_1 = True
                self.get_logger().info(f'Nytt mål sendt til {robot_id}: {target}')

    def send_path(self, start_pos, goal_pos, publisher):
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
                return True
        return False

    def clbk_map(self, msg):
        if self.prm is not None: return
        try:
            self.prm = PRM.from_occupancy_grid(msg, cell_size=1.5, points_per_cell=15, neighbors=12, max_edge_length=3.0, robot_radius=0.22)
            if hasattr(self.prm, 'free'):
                from scipy.ndimage import distance_transform_edt
                padded = __import__('numpy').pad(self.prm.free, 1, constant_values=False)
                self.prm.clearance_grid = distance_transform_edt(padded)[1:-1, 1:-1] * self.prm.resolution
            
            # Hent kartets dimensjoner og beregn senter for geofencing
            map_width = msg.info.width * msg.info.resolution
            map_height = msg.info.height * msg.info.resolution
            origin_x = msg.info.origin.position.x
            origin_y = msg.info.origin.position.y
            center_x = origin_x + (map_width / 2.0)
            
            # Opprett manageren med en grid-oppløsning på 1.5 meter
            self.manager = ExplorationManager(map_width, map_height, 1.5, origin_x, origin_y, center_x)
            
            self.publish_prm(msg.header.frame_id or 'map')
            self.get_logger().info('PRM og ExplorationManager er initialisert!')
        except Exception as e:
            self.get_logger().error(f'Kunne ikke bygge PRM: {e}')

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