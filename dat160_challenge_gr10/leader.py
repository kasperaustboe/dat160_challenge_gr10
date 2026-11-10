import math
import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import String, Bool, Empty
from tf2_ros import Buffer, TransformListener, TransformException
import tf2_geometry_msgs
from visualization_msgs.msg import Marker, MarkerArray

from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from scoring_interfaces.srv import SetMarkerPosition
from ros2_aruco_interfaces.msg import ArucoMarkers

from dat160_challenge_gr10.prm import PRM
from dat160_challenge_gr10.astar import AStar
from dat160_challenge_gr10.traffic import RouteReservations, in_y_zone


# Grupper etter de to startplassene i launch-filen.
EXPLORATION_SIDES = {'tb3_0': -1, 'tb3_3': -1, 'tb3_1': 1, 'tb3_4': 1}


class ExplorationManager:
    def __init__(self, map_width, map_height, resolution, origin_x, origin_y):
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        
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
        side = EXPLORATION_SIDES.get(robot_id)
        if side is None:
            return None
        valid_nodes = [n for n in prm_nodes if in_y_zone(n, side)]
        # Når hele sonen er besøkt kan roboten gjenbruke mål i samme sone.
        candidates = [n for n in valid_nodes if n not in self.visited_targets] or valid_nodes

        if not candidates:
            return None
            
        best_node = None
        min_cost = float('inf')

        for nx, ny in candidates:
            row, col = self._pos_to_grid(nx, ny)
            visits = self.visit_grid[row][col]
            dist = math.dist((current_x, current_y), (nx, ny))
            
            too_close = max(0.0, 1.5 - dist)
            cost = (visits * 150.0) + (dist * 0.5) + (too_close * 2.0)
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
        
        self.robot_ids = ["tb3_0", "tb3_1", "tb3_2", "tb3_3", "tb3_4"]
        self.helper_id = "tb3_2"
        self.declare_parameter('helper_spawn_x', -1.0)
        self.declare_parameter('helper_spawn_y', 0.0)
        self.helper_spawn = (float(self.get_parameter('helper_spawn_x').value),
                             float(self.get_parameter('helper_spawn_y').value))
        self.helper_at_spawn = True
        self.helper_route_origin = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.odom_frames = {}
        self.reported_waits = set()
        self.declare_parameter('rescue_radius', 0.8)
        self.declare_parameter('rescue_hold_seconds', 5.0)
        self.rescue_radius = float(self.get_parameter('rescue_radius').value)
        self.rescue_hold_seconds = float(self.get_parameter('rescue_hold_seconds').value)
        if not math.isfinite(self.rescue_radius) or self.rescue_radius < 0.6:
            raise ValueError('rescue_radius må være minst 0.6 meter')
        if not math.isfinite(self.rescue_hold_seconds) or self.rescue_hold_seconds < 5.0:
            raise ValueError('rescue_hold_seconds må være minst 5.0 sekunder')
        self.traffic = RouteReservations(clearance=0.75)
        self.planned_points = {}
        self.score_futures = {}
        self.rescue_cooldown_until = 0
        self.rescue = None
        self.held_robots = set()
        self.pending_reports = {}
        self.reports_in_flight = set()
        self.speed = {r: (float('inf'), float('inf')) for r in self.robot_ids}
        self.last_odom = {r: None for r in self.robot_ids}
        self.rescue_path_pubs = {
            r: self.create_publisher(Path, f'/{r}/rescue_path', 10)
            for r in self.robot_ids
        }
        self.hold_pubs = {
            r: self.create_publisher(Bool, f'/{r}/hold', 10)
            for r in self.robot_ids
        }
        self.pos = {r: None for r in self.robot_ids}
        self.is_exploring = {r: False for r in self.robot_ids}
        self.scan_wait_started = {}
        self.scan_left_pubs = {
            r: self.create_publisher(Empty, f'/{r}/scan_left', 10)
            for r in EXPLORATION_SIDES
        }

        self.observed_markers = set()
        self.observed_id4 = False
        self.pending_observations = {}
        self.found_qrs = set()
        self.total_targets = 5
        self.mission_finished = False
        self.robots_stopped = False

        self.score_client = self.create_client(SetMarkerPosition, '/set_marker_position')

        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.clbk_map, map_qos)
        self.mission_stop_pub = self.create_publisher(Bool, '/mission_stop', map_qos)
        
        for r_id in self.robot_ids:
            self.create_subscription(Odometry, f'/{r_id}/odom', lambda msg, rid=r_id: self.clbk_odom(msg, rid), qos_profile_sensor_data)
            self.create_subscription(String, f'/{r_id}/status', lambda msg, rid=r_id: self.clbk_status(msg, rid), 10)

        for r_id in self.robot_ids:
            self.create_subscription(
                ArucoMarkers, f'/{r_id}/aruco_markers',
                lambda msg, rid=r_id: self.clbk_aruco(msg, rid), 10
            )

        self.path_pubs = {r_id: self.create_publisher(Path, f'/{r_id}/path', 10) for r_id in self.robot_ids}
        
        # Publishere for visualisering i RViz
        self.visited_pub = self.create_publisher(MarkerArray, '/visited_nodes', 1)
        self.prm_marker_pub = self.create_publisher(MarkerArray, '/prm_graph', 1)
        self.route_marker_pub = self.create_publisher(MarkerArray, '/active_routes', 1)
        self.visited_points = []
        self.visited_node_ids = set()
        self.visited_dirty = True

        self.timer = self.create_timer(1.0, self.timer_callback)

    def debug_once(self, key, message):
        if key not in self.reported_waits:
            self.get_logger().debug(message)
            self.reported_waits.add(key)

    def clbk_odom(self, msg, r_id):
        frame = (msg.header.frame_id or f'{r_id}/odom').lstrip('/')
        self.odom_frames[r_id] = frame
        pose = msg.pose.pose
        try:
            if frame != 'map':
                transform = self.tf_buffer.lookup_transform(
                    'map', frame, rclpy.time.Time.from_msg(msg.header.stamp)
                )
                pose = tf2_geometry_msgs.do_transform_pose(pose, transform)
        except TransformException as ex:
            self.debug_once(
                ('odom_tf', r_id),
                f'{r_id}: får odometri, men mangler TF fra {frame} til map: {ex}'
            )
            return
        self.reported_waits.discard(('odom_tf', r_id))
        self.pos[r_id] = (pose.position.x, pose.position.y)
        if (r_id == self.helper_id and self.helper_route_origin is not None and
                math.dist(self.pos[r_id], self.helper_route_origin) >= self.traffic.clearance):
            self.helper_at_spawn = False
            self.helper_route_origin = None
        self.traffic.advance(r_id, self.pos[r_id])
        self.record_visited_nodes(self.pos[r_id])
        self.speed[r_id] = (
            math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y),
            abs(msg.twist.twist.angular.z),
        )
        self.last_odom[r_id] = self.get_clock().now().nanoseconds
        rescue = self.rescue
        if rescue and rescue['targets'] is not None and r_id in rescue['robots']:
            if not self.rescue_robot_ready(rescue, r_id, self.last_odom[r_id]):
                rescue['still_since'] = None
                self.pending_reports.pop(4, None)
        if self.manager:
            self.manager.update_robot_position(*self.pos[r_id])

    def clbk_status(self, msg, r_id):
        if msg.data == 'READY':
            self.traffic.release(r_id)
            self.is_exploring[r_id] = False
            self.held_robots.discard(r_id)
        elif msg.data == 'HOLDING':
            self.traffic.release(r_id)
            self.held_robots.add(r_id)
        elif msg.data == 'RESCUE_READY':
            rescue = self.rescue
            if (rescue and rescue['targets'] is not None and r_id in rescue['robots']
                    and r_id not in rescue['arrived']):
                rescue['arrived'].add(r_id)
                self.traffic.release(r_id)
                self.get_logger().info(f'{r_id} har kommet frem ved ID 4.')

    def clbk_aruco(self, msg, r_id):
        # Logg deteksjon selv om TF for bildet ennå ikke har kommet.
        for marker_id, camera_pose in zip(msg.marker_ids, msg.poses):
            marker_id = int(marker_id)
            if marker_id not in range(5):
                continue
            if marker_id == 4:
                if r_id == self.helper_id:
                    continue
                if not self.observed_id4:
                    self.observed_id4 = True
                    self.get_logger().info(f'{r_id} oppdaget ID 4, stor flamme.')
            elif marker_id not in self.observed_markers:
                self.observed_markers.add(marker_id)
                self.get_logger().info(f'{r_id} oppdaget markør ID {marker_id}.')
            if marker_id in self.found_qrs:
                continue
            # Samme optiske kameraramme som i deres opprinnelige marker_pose.py.
            # Noen versjoner av aruco_node publiserer et tomt frame_id.
            frame = (
                msg.header.frame_id or f'{r_id}/camera_rgb_optical_frame'
            ).lstrip('/')
            key = (r_id, marker_id)
            # Behold det opprinnelige tidsstempelet til TF er tilgjengelig.
            # Kontinuerlig overskriving med nyeste bilde kan ellers gi en
            # endeløs rekke ekstrapolasjonsfeil når TF ligger litt etter kamera.
            self.pending_observations.setdefault(
                key, (frame, msg.header.stamp, camera_pose)
            )
        self.process_marker_observations()

    def process_marker_observations(self):
        for key, observation in list(self.pending_observations.items()):
            r_id, marker_id = key
            if marker_id in self.found_qrs or (marker_id == 4 and self.rescue is not None):
                self.pending_observations.pop(key, None)
                continue
            frame, stamp, camera_pose = observation
            try:
                pose = camera_pose
                if frame != 'map':
                    transform = self.tf_buffer.lookup_transform(
                        'map', frame, rclpy.time.Time.from_msg(stamp)
                    )
                    pose = tf2_geometry_msgs.do_transform_pose(camera_pose, transform)
            except TransformException as ex:
                self.debug_once(
                    ('marker_tf', r_id, marker_id),
                    f'{r_id}: ID {marker_id} er observert, men venter på '
                    f'TF fra {frame} til map: {ex}'
                )
                # Et gammelt bilde kan ligge utenfor TF-bufferens historikk.
                # Gi da plass til en ny deteksjon med et ferskere tidsstempel.
                image_time = rclpy.time.Time.from_msg(stamp).nanoseconds
                if self.get_clock().now().nanoseconds - image_time > 3_000_000_000:
                    warning_key = ('id4_tf_expired', r_id)
                    if marker_id == 4 and warning_key not in self.reported_waits:
                        self.reported_waits.add(warning_key)
                        self.get_logger().warning(
                            f'{r_id}: ID 4 er lest, men kameraposisjonen kunne ikke '
                            f'omregnes fra {frame} til map. Venter på en ny observasjon.'
                        )
                    self.pending_observations.pop(key, None)
                continue
            if marker_id == 4:
                self.reported_waits.discard(('id4_tf_expired', r_id))
                now = self.get_clock().now().nanoseconds
                if (self.prm is None or self.pos[r_id] is None or self.pos[self.helper_id] is None
                        or now < self.rescue_cooldown_until):
                    # Behold posisjonen i map før transformasjonen for bildet blir for gammel.
                    self.pending_observations[key] = ('map', stamp, pose)
                    continue
            self.pending_observations.pop(key, None)
            self.reported_waits.discard(('marker_tf', r_id, marker_id))
            position = Point(
                x=float(pose.position.x), y=float(pose.position.y),
                z=float(pose.position.z)
            )
            if marker_id == 4:
                q = pose.orientation
                # Markørens lokale z-akse er normalen til markørflaten.
                normal = (2 * (q.x * q.z + q.w * q.y),
                          2 * (q.y * q.z - q.w * q.x))
                self.start_rescue(r_id, position, normal)
            else:
                self.register_marker(marker_id, position)

    def register_marker(self, marker_id, position):
        if marker_id not in range(5) or marker_id in self.found_qrs:
            return
        if marker_id not in self.reports_in_flight:
            self.pending_reports[marker_id] = position

    def call_scoring_service(self, marker_id, position):
        if not self.score_client.service_is_ready():
            return
        request = SetMarkerPosition.Request()
        request.marker_id = int(marker_id)
        request.marker_position = position
        self.reports_in_flight.add(marker_id)
        future = self.score_client.call_async(request)
        self.score_futures[marker_id] = future
        future.add_done_callback(lambda f: self.clbk_score_response(f, marker_id))

    def clbk_score_response(self, future, marker_id):
        # Et sent svar fra et avbrutt forsøk må ikke frigi et nytt redningspar.
        if self.score_futures.get(marker_id) is not future:
            return
        self.score_futures.pop(marker_id, None)
        self.reports_in_flight.discard(marker_id)
        try:
            response = future.result()
            if not response.accepted:
                # La timeren prøve igjen innen den eksisterende tidsgrensen.
                return
        except Exception as ex:
            if marker_id == 4:
                self.release_rescue(f'Feil ved scoring av ID 4: {ex}.')
            else:
                self.get_logger().error(f'Feil ved scoring: {ex}')
            return
        self.pending_reports.pop(marker_id, None)
        self.found_qrs.add(marker_id)
        if marker_id == 4 and self.rescue is not None:
            # Scoring sjekker nærhet i timeren etter at tjenesten har tatt imot ID 4.
            self.rescue['release_after'] = self.get_clock().now().nanoseconds + 1_500_000_000
        if len(self.found_qrs) >= self.total_targets:
            self.mission_finished = True
            self.stop_all_robots()

    def stop_all_robots(self):
        if self.robots_stopped:
            return
        self.robots_stopped = True
        self.mission_stop_pub.publish(Bool(data=True))
        for rid in self.robot_ids:
            self.traffic.release(rid)
        self.get_logger().info('Oppdrag fullført. Stopper alle roboter.')

    def start_rescue(self, finder_id, position, normal=None):
        now = self.get_clock().now().nanoseconds
        if (self.robots_stopped or finder_id == self.helper_id or now < self.rescue_cooldown_until
                or self.pos[finder_id] is None):
            return
        observer = self.pos[finder_id]
        view = (observer[0] - position.x, observer[1] - position.y)
        if normal is None or math.hypot(*normal) < 0.1:
            normal = view
        length = math.hypot(*normal)
        if length < 0.001:
            return
        normal = (normal[0] / length, normal[1] / length)
        if normal[0] * view[0] + normal[1] * view[1] < 0:
            normal = (-normal[0], -normal[1])
        robots = (finder_id, self.helper_id)
        if any(self.pos[r] is None for r in robots):
            return
        self.rescue = {
            'robots': robots, 'position': position,
            'targets': None, 'still_since': None,
            'normal': normal, 'plan_started': now, 'report_wait_since': None,
            'arrived': set(), 'planner': None, 'release_after': None,
        }
        for rid in robots:
            self.scan_wait_started.pop(rid, None)
            self.held_robots.discard(rid)
            self.hold_pubs[rid].publish(Bool(data=True))
            self.is_exploring[rid] = True
        self.get_logger().debug(
            f'STOR FLAMME ID 4: {finder_id} stopper. {self.helper_id} sendes fra spawn.'
        )

    def release_rescue(self, reason=None):
        if self.rescue is not None:
            for rid in self.rescue['robots']:
                self.traffic.release(rid)
                if not self.robots_stopped:
                    self.hold_pubs[rid].publish(Bool(data=False))
                    # READY bekrefter at handler har behandlet frigivelsen.
                    self.is_exploring[rid] = True
            self.rescue = None
        self.pending_reports.pop(4, None)
        self.reports_in_flight.discard(4)
        self.score_futures.pop(4, None)
        if reason:
            self.get_logger().warning(reason)
            self.rescue_cooldown_until = self.get_clock().now().nanoseconds + 20_000_000_000

    def flame_visible(self, point, rescue):
        flame = (rescue['position'].x, rescue['position'].y)
        normal = rescue['normal']
        # Samme side av markørflaten som kameraet som observerte den.
        if ((point[0] - flame[0]) * normal[0] +
                (point[1] - flame[1]) * normal[1]) <= 0.10:
            return False
        # Bruk råkartet for sikt, ikke robotens oppblåste sikkerhetsområde.
        # Markørens egen celle kan være en veggcelle; resten må være fritt.
        end_cell = self.prm.world_to_grid(*flame)
        steps = max(1, math.ceil(math.dist(point, flame) / (self.prm.resolution / 3)))
        for i in range(steps + 1):
            t = i / steps
            x = point[0] + t * (flame[0] - point[0])
            y = point[1] + t * (flame[1] - point[1])
            gx, gy = self.prm.world_to_grid(x, y)
            if not (0 <= gx < self.prm.width and 0 <= gy < self.prm.height):
                return False
            if (gx, gy) == end_cell:
                continue
            value = self.prm.grid[gy, gx]
            if value < 0 or value >= self.prm.obstacle_threshold:
                return False
        return True

    def planning_positions(self, robot_id):
        if robot_id == self.helper_id or not self.helper_at_spawn:
            return self.pos
        # Sperr plassen fra oppstart, også før hjelpers odometri og TF er tilgjengelig.
        return {**self.pos, 'helper_spawn': self.helper_spawn}

    def make_path(self, robot_id, target, extra_routes=None, *, rescue=False):
        points = self.traffic.plan(
            self.prm, AStar, robot_id, self.pos[robot_id], target,
            self.planning_positions(robot_id), extra_routes,
            zone_side=None if rescue else EXPLORATION_SIDES.get(robot_id),
        )
        if not points:
            return None
        return self.path_from_points(robot_id, points)

    def path_from_points(self, robot_id, points):
        message = Path()
        message.header.frame_id = 'map'
        message.header.stamp = self.get_clock().now().to_msg()
        for x, y in points:
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            message.poses.append(pose)
        result = self.path_to_odom(message, robot_id)
        if result is not None:
            self.planned_points[robot_id] = points
        return result

    def path_to_odom(self, message, robot_id):
        # Robot_handler sammenligner med rå odometri. Send derfor rutepunktene
        # i akkurat den robotens odometriramme, mens planleggingen bruker map.
        frame = self.odom_frames.get(robot_id)
        if frame is None:
            return None
        try:
            transform = None
            if frame != 'map':
                transform = self.tf_buffer.lookup_transform(
                    frame, 'map', rclpy.time.Time()
                )
            message.header.frame_id = frame
            message.header.stamp = self.get_clock().now().to_msg()
            for pose in message.poses:
                if transform is not None:
                    pose.pose = tf2_geometry_msgs.do_transform_pose(pose.pose, transform)
                pose.header = message.header
            return message
        except TransformException as ex:
            self.debug_once(
                ('path_tf', robot_id),
                f'{robot_id}: kan ikke omregne ruten fra map til {frame}: {ex}'
            )
            return None

    def rescue_route_options(self, rescue):
        finder, helper = rescue['robots']
        flame = (rescue['position'].x, rescue['position'].y)
        outer = self.rescue_radius - 0.15
        candidates = [p for p in self.prm.nodes if 0.40 <= math.dist(p, flame) <= outer]
        for radius in (0.45, 0.55, outer):
            if not 0.40 <= radius <= outer:
                continue
            for i in range(24):
                angle = i * math.tau / 24
                point = (flame[0] + radius * math.cos(angle),
                         flame[1] + radius * math.sin(angle))
                if self.prm.is_free(*point):
                    candidates.append(point)
        candidates = [p for p in candidates if self.flame_visible(p, rescue)]
        # Velg korteste kjørbare hjelperrute først. Finneren må få plass ved siden av.
        finder_candidates = sorted(candidates, key=lambda p: math.dist(p, self.pos[finder]))
        helper_routes = self.traffic.shortest_routes(
            self.prm, helper, self.pos[helper], candidates, self.planning_positions(helper)
        )
        for second_points in helper_routes:
            second = second_points[-1]
            second_path = self.path_from_points(helper, second_points)
            if second_path is None:
                return
            for first in finder_candidates:
                if math.dist(first, second) < self.traffic.clearance:
                    continue
                first_path = self.make_path(
                    finder, first, extra_routes={helper: second_points}, rescue=True
                )
                if first_path is None:
                    yield None  # Gi timeren en pause etter hvert rutesøk.
                    continue
                first_points = self.planned_points[finder]
                yield ((first, second), (first_path, second_path), (first_points, second_points))

        # Hvis hjelperens korteste ruter sperrer finneren, prøv å legge finnerens rute først.
        for first in finder_candidates:
            first_path = self.make_path(finder, first, rescue=True)
            yield None
            if first_path is None:
                continue
            first_points = self.planned_points[finder]
            helper_routes = self.traffic.shortest_routes(
                self.prm, helper, self.pos[helper], candidates, self.planning_positions(helper),
                extra_routes={finder: first_points},
            )
            for second_points in helper_routes:
                second_path = self.path_from_points(helper, second_points)
                if second_path is None:
                    return
                yield ((first, second_points[-1]), (first_path, second_path),
                       (first_points, second_points))

    def rescue_robot_ready(self, rescue, robot_id, now):
        target = rescue['targets'][rescue['robots'].index(robot_id)]
        flame = (rescue['position'].x, rescue['position'].y)
        position = self.pos[robot_id]
        stamp = self.last_odom[robot_id]
        return (robot_id in rescue['arrived'] and stamp is not None
                and 0 <= now - stamp < 1_000_000_000 and position is not None
                and math.dist(position, target) <= 0.15
                and math.dist(position, flame) <= self.rescue_radius
                and self.flame_visible(position, rescue)
                and self.speed[robot_id][0] < 0.02 and self.speed[robot_id][1] < 0.03)

    def process_rescue(self):
        rescue = self.rescue
        if rescue is None:
            return
        robots = rescue['robots']
        now = self.get_clock().now().nanoseconds
        if rescue['release_after'] is not None:
            if now >= rescue['release_after']:
                self.release_rescue()
            return
        if (rescue['report_wait_since'] is not None and
                now - rescue['report_wait_since'] >= 10_000_000_000):
            self.release_rescue()
            self.rescue_cooldown_until = now + 20_000_000_000
            return
        if rescue['targets'] is None and now - rescue['plan_started'] > 20_000_000_000:
            self.release_rescue('ID 4: fant ikke to ledige redningsruter. Robotene frigis.')
            return
        if rescue['targets'] is None:
            # Vent på stoppbekreftelse før en ny rute sendes på et annet topic.
            if not all(r in self.held_robots for r in robots):
                for rid in robots:
                    if rid not in self.held_robots:
                        self.hold_pubs[rid].publish(Bool(data=True))
                return
            if any(self.rescue_path_pubs[r].get_subscription_count() == 0 for r in robots):
                return
            if rescue['planner'] is None:
                rescue['planner'] = self.rescue_route_options(rescue)
            for _ in range(3):
                try:
                    option = next(rescue['planner'])
                except StopIteration:
                    rescue['planner'] = None
                    return
                if option is None:
                    continue
                targets, paths, routes = option
                # Andre roboter kan ha flyttet seg siden første rutesøk.
                # Sjekk reservasjonene igjen før paret får kjøre.
                for robot_id, points in zip(robots, routes):
                    occupied = self.traffic.segments(robot_id, self.planning_positions(robot_id))
                    if (math.dist(self.pos[robot_id], points[0]) > 0.10 or
                            any(not self.traffic.clear_segment(a, b, occupied)
                                for a, b in zip(points, points[1:]))):
                        break
                else:
                    rescue['targets'] = targets
                    rescue['planner'] = None
                    if self.helper_at_spawn and self.helper_route_origin is None:
                        self.helper_route_origin = self.pos[self.helper_id]
                    for robot_id, points, path in zip(robots, routes, paths):
                        self.traffic.reserve(robot_id, points)
                        self.rescue_path_pubs[robot_id].publish(path)
                    return
            return
        if not all(self.rescue_robot_ready(rescue, r, now) for r in robots):
            rescue['still_since'] = None
            self.pending_reports.pop(4, None)
            return
        if rescue['still_since'] is None:
            rescue['still_since'] = now
            self.get_logger().debug(
                f'Begge roboter står ved ID 4. Venter {self.rescue_hold_seconds:.1f} sekunder.'
            )
        elapsed = (now - rescue['still_since']) / 1e9
        if elapsed >= self.rescue_hold_seconds:
            if rescue['report_wait_since'] is None:
                rescue['report_wait_since'] = now
            self.register_marker(4, rescue['position'])

    def timer_callback(self):
        if self.visited_dirty:
            self.publish_visited_nodes()
            self.visited_dirty = False
        self.publish_active_routes()
        if self.mission_finished:
            return
        self.process_marker_observations()
        self.process_rescue()
        for marker_id, position in list(self.pending_reports.items()):
            if marker_id not in self.reports_in_flight:
                self.call_scoring_service(marker_id, position)
        if not self.robots_stopped and self.prm is not None and self.manager is not None:
            reserved = self.rescue['robots'] if self.rescue else ()
            for r_id in self.robot_ids:
                if r_id == self.helper_id or r_id in reserved:
                    continue
                if self.pos[r_id] is None:
                    if r_id in ('tb3_0', 'tb3_1'):
                        self.debug_once(
                            ('position', r_id),
                            f'{r_id}: venter på robotposisjon i map. Sjekk odometri og TF.'
                        )
                    continue
                self.reported_waits.discard(('position', r_id))
                if self.is_exploring[r_id]:
                    continue
                if self.path_pubs[r_id].get_subscription_count() == 0:
                    self.debug_once(
                        ('subscriber', r_id),
                        f'{r_id}: venter på robot_handler som abonnent på /{r_id}/path.'
                    )
                    continue
                self.reported_waits.discard(('subscriber', r_id))
                self.assign_new_target(r_id, self.pos[r_id], self.path_pubs[r_id])
                if not self.is_exploring[r_id]:
                    self.request_left_scan(r_id)

    def request_left_scan(self, robot_id):
        now = self.get_clock().now().nanoseconds
        started = self.scan_wait_started.setdefault(robot_id, now)
        if started is None or now - started < 3_000_000_000:
            return
        publisher = self.scan_left_pubs[robot_id]
        if publisher.get_subscription_count() > 0:
            publisher.publish(Empty())
            # Én skann per venteperiode. Neste rute nullstiller ventingen.
            self.scan_wait_started[robot_id] = None

    def assign_new_target(self, robot_id, start_pos, publisher):
        if publisher.get_subscription_count() == 0:
            return
        if not self.prm.is_free(*start_pos):
            self.debug_once(
                ('blocked_start', robot_id),
                f'{robot_id}: posisjon {start_pos} er utenfor ledig PRM-område. '
                'Sjekk map/odom, spawnposisjon og robot_radius.'
            )
            return
        self.reported_waits.discard(('blocked_start', robot_id))
        target = self.manager.get_next_target(robot_id, self.prm.nodes, start_pos[0], start_pos[1])
        if target:
            if self.send_path(start_pos, target, publisher, robot_id):
                self.is_exploring[robot_id] = True
                self.scan_wait_started.pop(robot_id, None)
                self.reported_waits.discard(('route', robot_id))
                return
            self.debug_once(
                ('route', robot_id),
                f'{robot_id}: fant ikke sendbar rute fra {start_pos} til {target}. '
                'Sjekk PRM-forbindelser og TF.'
            )
        else:
            self.debug_once(('target', robot_id), f'{robot_id}: ingen mål tilgjengelig i PRM.')

    def send_path(self, start_pos, goal_pos, publisher, robot_id):
        if publisher.get_subscription_count() == 0:
            return False
        target = goal_pos if isinstance(goal_pos, tuple) else (goal_pos.x, goal_pos.y)
        path_msg = self.make_path(robot_id, target)
        if path_msg is None:
            return False
        # make_path legger robotens nåværende posisjon først. Den er ikke et stopp.
        points = self.planned_points[robot_id]
        path_msg.poses = path_msg.poses[1:13]
        if not path_msg.poses:
            return False
        self.traffic.reserve(robot_id, points[:1 + len(path_msg.poses)])
        publisher.publish(path_msg)
        return True

    def record_visited_nodes(self, position):
        if self.prm is None:
            return
        # Grønt betyr faktisk besøkt, ikke valgt mål eller planlagt rute.
        for node_id, point in enumerate(self.prm.nodes):
            if node_id in self.visited_node_ids:
                continue
            if (math.dist(position, point) <= 0.10 and
                    self.prm.line_is_free(position, point)):
                self.visited_node_ids.add(node_id)
                self.visited_points.append(point)
                self.visited_dirty = True

    def publish_active_routes(self):
        markers = MarkerArray()
        visited = set(self.visited_points)
        for marker_id, robot_id in enumerate(self.robot_ids):
            points = self.traffic.routes.get(robot_id, ())
            line = Marker()
            line.header.frame_id = 'map'
            line.ns = 'active_routes'
            line.id = marker_id
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD if len(points) >= 2 else Marker.DELETE
            line.pose.orientation.w = 1.0
            line.scale.x = 0.04
            line.color.r = line.color.g = 1.0
            line.color.a = 1.0
            for x, y in points:
                line.points.append(Point(x=float(x), y=float(y), z=0.08))
            markers.markers.append(line)

            nodes = Marker()
            nodes.header.frame_id = 'map'
            nodes.ns = 'reserved_nodes'
            nodes.id = marker_id
            nodes.type = Marker.SPHERE_LIST
            nodes.action = Marker.ADD if points else Marker.DELETE
            nodes.pose.orientation.w = 1.0
            nodes.scale.x = nodes.scale.y = nodes.scale.z = 0.18
            nodes.color.r = nodes.color.g = 1.0
            nodes.color.a = 1.0
            # Første punkt er robotens posisjon; resten er gjenstående veipunkter.
            for point in points[1:]:
                if point not in visited:
                    nodes.points.append(Point(x=float(point[0]), y=float(point[1]), z=0.08))
            markers.markers.append(nodes)
        self.route_marker_pub.publish(markers)

    def publish_visited_nodes(self):
        markers = MarkerArray()
        node_marker = Marker()
        node_marker.header.frame_id = 'map'
        node_marker.ns = 'visited_nodes'
        node_marker.id = 10
        node_marker.pose.orientation.w = 1.0
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
            # W1: tb3_3 og tb3_4 deler startplassene til tb3_0 og tb3_1.
            spawn_points = [(0.0, -1.0), (0.0, 1.0), self.helper_spawn]
            self.prm = PRM.from_occupancy_grid(
                msg, cell_size=1.5, points_per_cell=15, neighbors=12,
                max_edge_length=3.0, robot_radius=0.32, reachable_from=spawn_points,
            )
            map_width = msg.info.width * msg.info.resolution
            map_height = msg.info.height * msg.info.resolution
            origin_x = msg.info.origin.position.x
            origin_y = msg.info.origin.position.y
            self.manager = ExplorationManager(map_width, map_height, 1.5, origin_x, origin_y)
            self.get_logger().debug('PRM klar. tb3_0, tb3_1, tb3_3 og tb3_4 utforsker; tb3_2 er hjelper.')
            
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
