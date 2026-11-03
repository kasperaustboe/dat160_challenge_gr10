import math
import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from scoring_interfaces.srv import SetMarkerPosition
from ros2_aruco_interfaces.msg import ArucoMarkers

from dat160_challenge_gr10.prm import PRM
from dat160_challenge_gr10.astar import AStar


class ExplorationManager:
    def __init__(self, map_width, map_height, resolution, origin_x, origin_y, center_x, center_y):
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.center_x = center_x
        self.center_y = center_y
        
        self.cols = int(map_width / resolution)
        self.rows = int(map_height / resolution)
        self.visit_grid = np.zeros((self.rows, self.cols), dtype=int)
        self.visited_targets = set()

    def _pos_to_grid(self, x, y):
        adj_x, adj_y = x - self.origin_x, y - self.origin_y
        col = max(0, min(int(adj_x / self.resolution), self.cols - 1))
        row = max(0, min(int(adj_y / self.resolution), self.rows - 1))
        return row, col

    def update_robot_position(self, x, y):
        row, col = self._pos_to_grid(x, y)
        self.visit_grid[row][col] += 1

    def get_next_target(self, robot_id, prm_nodes, current_x, current_y):
        valid_nodes = []
        
        # Strenge soner: tb3_0 på venstre side, tb3_1 på høyre side
        for node_x, node_y in prm_nodes:
            if robot_id == "tb3_0" and node_x < self.center_x:
                valid_nodes.append((node_x, node_y))
            elif robot_id == "tb3_1" and node_x >= self.center_x:
                valid_nodes.append((node_x, node_y))

        # Filtrer bort allerede besøkte mål i sonen
        candidates = [n for n in valid_nodes if n not in self.visited_targets]
        
        # Backup hvis sonen er tom
        if not candidates:
            candidates = [n for n in prm_nodes if n not in self.visited_targets]
            if not candidates:
                candidates = prm_nodes

        if not candidates:
            return None
            
        best_node = None
        min_cost = float('inf')

        for nx, ny in candidates:
            row, col = self._pos_to_grid(nx, ny)
            visits = self.visit_grid[row][col]
            dist = math.dist((current_x, current_y), (nx, ny))
            
            cost = (visits * 150.0) + (dist * 0.5)
            if cost < min_cost:
                min_cost = cost
                best_node = (nx, ny)
                
        if best_node:
            self.visited_targets.add(best_node)
            
        return best_node


class LeaderClass(Node):
    def __init__(self):
        super().__init__('RobotLeaderNode')
        self.prm = None
        self.manager = None
        
        self.robot_ids = ["tb3_0", "tb3_1"]
        self.pos = {r: None for r in self.robot_ids}
        self.is_exploring = {r: False for r in self.robot_ids}

        self.found_qrs = set()
        self.total_targets = 5
        self.mission_finished = False

        self.score_client = self.create_client(SetMarkerPosition, 'set_marker_position')

        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.clbk_map, map_qos)
        
        for r_id in self.robot_ids:
            self.create_subscription(Odometry, f'/{r_id}/odom', lambda msg, rid=r_id: self.clbk_odom(msg, rid), qos_profile_sensor_data)
            self.create_subscription(String, f'/{r_id}/status', lambda msg, rid=r_id: self.clbk_status(msg, rid), 10)

        self.create_subscription(ArucoMarkers, '/aruco_markers', self.clbk_aruco, 10)
        for r_id in self.robot_ids:
            self.create_subscription(ArucoMarkers, f'/{r_id}/aruco_markers', self.clbk_aruco, 10)

        self.path_pubs = {r_id: self.create_publisher(Path, f'/{r_id}/path', 10) for r_id in self.robot_ids}
        
        # Publishere for visualisering i RViz
        self.visited_pub = self.create_publisher(MarkerArray, '/visited_nodes', 1)
        self.prm_marker_pub = self.create_publisher(MarkerArray, '/prm_graph', 1)
        self.visited_points = []

        self.timer = self.create_timer(1.0, self.timer_callback)

    def clbk_odom(self, msg, r_id):
        self.pos[r_id] = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if self.manager:
            self.manager.update_robot_position(*self.pos[r_id])

    def clbk_status(self, msg, r_id):
        if msg.data == "READY":
            self.is_exploring[r_id] = False

    def clbk_aruco(self, msg):
        for i in range(len(msg.marker_ids)):
            self.register_marker(msg.marker_ids[i], Point(x=msg.poses[i].position.x, y=msg.poses[i].position.y, z=msg.poses[i].position.z))

    def register_marker(self, marker_id, position):
        gyldige_maler = [0, 1, 2, 3, 4]
        if marker_id not in self.found_qrs:
            if marker_id not in gyldige_maler:
                return
            self.found_qrs.add(marker_id)
            self.get_logger().info(f'--- FANT EKTE MARKØR! ID: {marker_id} ({len(self.found_qrs)}/{self.total_targets}) ---')
            self.call_scoring_service(marker_id, position)

            if len(self.found_qrs) >= self.total_targets:
                self.mission_finished = True
                self.get_logger().info('*** ALLE 5 MARKØRER ER REGISTRERT - OPPDRAG FULLFØRT! ***')

    def call_scoring_service(self, marker_id, position):
        request = SetMarkerPosition.Request()
        request.marker_id = int(marker_id)
        request.marker_position = position
        future = self.score_client.call_async(request)
        future.add_done_callback(lambda f: self.clbk_score_response(f, marker_id))

    def clbk_score_response(self, future, marker_id):
        try:
            response = future.result()
            if response.accepted:
                self.get_logger().info(f'Scoring GODKJENTE ID {marker_id}!')
        except Exception as e:
            self.get_logger().error(f'Feil ved scoring: {e}')

    def timer_callback(self):
        if self.mission_finished:
            return  

        if self.prm is not None and self.manager is not None:
            for r_id in self.robot_ids:
                if self.pos[r_id] and not self.is_exploring[r_id]:
                    self.assign_new_target(r_id, self.pos[r_id], self.path_pubs[r_id])

    def assign_new_target(self, robot_id, start_pos, publisher):
        target = self.manager.get_next_target(robot_id, self.prm.nodes, start_pos[0], start_pos[1])
        if target:
            if target not in self.visited_points:
                self.visited_points.append(target)
                self.publish_visited_nodes()

            if self.send_path(start_pos, target, publisher):
                self.is_exploring[robot_id] = True
                return

    def send_path(self, start_pos, goal_pos, publisher):
        gx, gy = goal_pos if isinstance(goal_pos, tuple) else (goal_pos.x, goal_pos.y)
        start_node = self.prm.nearest_node(start_pos[0], start_pos[1], visible_only=False)
        goal_node = self.prm.nearest_node(gx, gy, visible_only=False)

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

    def publish_visited_nodes(self):
        markers = MarkerArray()
        node_marker = Marker()
        node_marker.header.frame_id = 'map'
        node_marker.ns = 'visited_nodes'
        node_marker.id = 10
        node_marker.type = Marker.SPHERE_LIST
        node_marker.action = Marker.ADD
        node_marker.scale.x = node_marker.scale.y = node_marker.scale.z = 0.2
        node_marker.color.g = 1.0
        node_marker.color.a = 1.0
        for (x, y) in self.visited_points:
            node_marker.points.append(Point(x=float(x), y=float(y), z=0.1))
        markers.markers.append(node_marker)
        self.visited_pub.publish(markers)

    def publish_prm_graph(self):
        """Tegner PRM-nodene som røde prikker i RViz"""
        if self.prm is None:
            return
        markers = MarkerArray()
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.ns = 'prm_nodes'
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = 0.15
        marker.color.r = 1.0
        marker.color.a = 0.8
        for (x, y) in self.prm.nodes:
            marker.points.append(Point(x=float(x), y=float(y), z=0.05))
        markers.markers.append(marker)
        self.prm_marker_pub.publish(markers)

    def clbk_map(self, msg):
        if self.prm is not None:
            return
        try:
            self.prm = PRM.from_occupancy_grid(msg, cell_size=1.5, points_per_cell=15, neighbors=12, max_edge_length=3.0, robot_radius=0.22)
            map_width = msg.info.width * msg.info.resolution
            map_height = msg.info.height * msg.info.resolution
            origin_x = msg.info.origin.position.x
            origin_y = msg.info.origin.position.y
            center_x = origin_x + (map_width / 2.0)
            center_y = origin_y + (map_height / 2.0)
            
            self.manager = ExplorationManager(map_width, map_height, 1.5, origin_x, origin_y, center_x, center_y)
            self.get_logger().info('PRM og venstre/høyre-soner for tb3_0 og tb3_1 er initialisert!')
            
            # Publiser de røde PRM-nodene med en gang kartet er klart
            self.publish_prm_graph()
        except Exception as e:
            self.get_logger().error(f'Feil ved PRM-init: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = LeaderClass()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()