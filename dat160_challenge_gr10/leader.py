import math
from dataclasses import dataclass, field

import numpy as np
import rclpy
import tf2_geometry_msgs
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import String, Bool, Empty
from tf2_ros import Buffer, TransformListener, TransformException
from visualization_msgs.msg import Marker, MarkerArray

from ros2_aruco_interfaces.msg import ArucoMarkers
from scoring_interfaces.srv import SetMarkerPosition

from dat160_challenge_gr10.astar import AStar
from dat160_challenge_gr10.prm import PRM
from dat160_challenge_gr10.traffic import RouteReservations, in_y_zone


ROBOT_IDS = ['tb3_0', 'tb3_1', 'tb3_2', 'tb3_3', 'tb3_4']
HELPER_ID = 'tb3_2'
FLAME_ID = 4          # Stor flamme: krever to roboter ved markøren.
TOTAL_TARGETS = 5     # Markør-ID 0-4

# Grupper etter de to startplassene i launch-filen (-1: y < 0, 1: y > 0).
EXPLORATION_SIDES = {'tb3_0': -1, 'tb3_3': -1, 'tb3_1': 1, 'tb3_4': 1}

SEC = 1_000_000_000  # nanosekunder


def seconds(value):
    return int(value * SEC)


@dataclass
class Rescue:
    """Tilstand for en pågående redning ved ID 4."""
    robots: tuple
    position: Point
    normal: tuple             # Enhetsvektor ut fra markørflaten, mot kameraet
    retry_at: int
    warn_after: int
    targets: dict = field(default_factory=dict)   # robot -> stoppunkt
    arrived: set = field(default_factory=set)
    still_since: int = None
    report_wait_since: int = None
    release_after: int = None

    @property
    def flame(self):
        return (self.position.x, self.position.y)


class ExplorationManager:
    """Velger utforskningsmål som er lite besøkt og i nærheten."""

    def __init__(self, map_width, map_height, resolution, origin_x, origin_y):
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.cols = int(map_width / resolution)
        self.rows = int(map_height / resolution)
        self.visit_grid = np.zeros((self.rows, self.cols), dtype=int)
        self.visited_targets = set()

    def _pos_to_grid(self, x, y):
        col = int((x - self.origin_x) / self.resolution)
        row = int((y - self.origin_y) / self.resolution)
        return max(0, min(row, self.rows - 1)), max(0, min(col, self.cols - 1))

    def update_robot_position(self, x, y):
        self.visit_grid[self._pos_to_grid(x, y)] += 1

    def get_next_target(self, robot_id, prm_nodes, current_x, current_y):
        side = EXPLORATION_SIDES.get(robot_id)
        if side is None:
            return None
        valid_nodes = [n for n in prm_nodes if in_y_zone(n, side)]
        # Når hele sonen er besøkt kan roboten gjenbruke mål i samme sone.
        candidates = [n for n in valid_nodes if n not in self.visited_targets] or valid_nodes
        if not candidates:
            return None

        def cost(node):
            visits = self.visit_grid[self._pos_to_grid(*node)]
            return visits * 150.0 + math.dist((current_x, current_y), node) * 0.5

        best_node = min(candidates, key=cost)
        self.visited_targets.add(best_node)
        return best_node


def sphere_list(ns, marker_id, size, color, points, z):
    """Lag en SPHERE_LIST-markør for RViz. color er (r, g, b, a)."""
    marker = Marker()
    marker.header.frame_id = 'map'
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.SPHERE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = marker.scale.y = marker.scale.z = size
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
    marker.points = [Point(x=float(x), y=float(y), z=z) for x, y in points]
    return marker


class LeaderClass(Node):
    def __init__(self):
        super().__init__('RobotLeaderNode')
        self.robot_ids = ROBOT_IDS
        self.helper_id = HELPER_ID

        # --- Parametere ---
        self.declare_parameter('helper_spawn_x', -1.0)
        self.declare_parameter('helper_spawn_y', 0.0)
        self.declare_parameter('rescue_radius', 1.5)
        self.declare_parameter('rescue_hold_seconds', 5.0)
        self.helper_spawn = (float(self.get_parameter('helper_spawn_x').value),
                             float(self.get_parameter('helper_spawn_y').value))
        self.rescue_radius = float(self.get_parameter('rescue_radius').value)
        self.rescue_hold_seconds = float(self.get_parameter('rescue_hold_seconds').value)
        if not math.isfinite(self.rescue_radius) or self.rescue_radius < 0.6:
            raise ValueError('rescue_radius må være minst 0.6 meter')
        if not math.isfinite(self.rescue_hold_seconds) or self.rescue_hold_seconds < 5.0:
            raise ValueError('rescue_hold_seconds må være minst 5.0 sekunder')

        # --- Planlegging ---
        self.prm = None
        self.manager = None
        self.traffic = RouteReservations(clearance=0.75)

        # --- Robottilstand ---
        self.pos = {r: None for r in ROBOT_IDS}
        self.speed = {r: (math.inf, math.inf) for r in ROBOT_IDS}  # (lineær, vinkel)
        self.last_odom = {r: None for r in ROBOT_IDS}
        self.odom_frames = {}
        self.is_exploring = {r: False for r in ROBOT_IDS}
        self.held_robots = set()
        self.scan_wait_started = {}
        self.helper_at_spawn = True
        self.helper_route_origin = None

        # --- Markører og scoring ---
        self.observed_markers = set()
        self.observed_id4 = False
        self.pending_observations = {}  # (robot, marker) -> (frame, stamp, pose)
        self.pending_reports = {}       # marker -> posisjon
        self.reports_in_flight = set()
        self.score_futures = {}
        self.found_qrs = set()
        self.total_targets = TOTAL_TARGETS
        self.rescue = None
        self.rescue_cooldown_until = 0
        self.mission_finished = False
        self.robots_stopped = False
        self.reported_waits = set()     # Nøkler for meldinger som bare logges én gang

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.score_client = self.create_client(SetMarkerPosition, '/set_marker_position')

        # --- Topics ---
        latched_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.clbk_map, latched_qos)
        self.mission_stop_pub = self.create_publisher(Bool, '/mission_stop', latched_qos)

        for r in ROBOT_IDS:
            self.create_subscription(Odometry, f'/{r}/odom',
                                     lambda msg, rid=r: self.clbk_odom(msg, rid), qos_profile_sensor_data)
            self.create_subscription(String, f'/{r}/status',
                                     lambda msg, rid=r: self.clbk_status(msg, rid), 10)
            self.create_subscription(ArucoMarkers, f'/{r}/aruco_markers',
                                     lambda msg, rid=r: self.clbk_aruco(msg, rid), 10)

        self.path_pubs = {r: self.create_publisher(Path, f'/{r}/path', 10) for r in ROBOT_IDS}
        self.rescue_path_pubs = {r: self.create_publisher(Path, f'/{r}/rescue_path', 10) for r in ROBOT_IDS}
        self.hold_pubs = {r: self.create_publisher(Bool, f'/{r}/hold', 10) for r in ROBOT_IDS}
        self.scan_left_pubs = {r: self.create_publisher(Empty, f'/{r}/scan_left', 10)
                               for r in EXPLORATION_SIDES}

        # Visualisering i RViz
        self.visited_pub = self.create_publisher(MarkerArray, '/visited_nodes', 1)
        self.prm_marker_pub = self.create_publisher(MarkerArray, '/prm_graph', 1)
        self.route_marker_pub = self.create_publisher(MarkerArray, '/active_routes', 1)
        self.visited_points = []
        self.visited_node_ids = set()
        self.visited_dirty = True

        self.timer = self.create_timer(1.0, self.timer_callback)

    # --- Hjelpefunksjoner ---

    def now(self):
        return self.get_clock().now().nanoseconds

    def debug_once(self, key, message):
        if key not in self.reported_waits:
            self.get_logger().debug(message)
            self.reported_waits.add(key)

    def to_map(self, pose, frame, stamp):
        """Transformer en pose til map. Kaster TransformException ved manglende TF."""
        if frame == 'map':
            return pose
        transform = self.tf_buffer.lookup_transform('map', frame, rclpy.time.Time.from_msg(stamp))
        return tf2_geometry_msgs.do_transform_pose(pose, transform)

    # --- Callbacks ---

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
            info = msg.info
            self.manager = ExplorationManager(
                info.width * info.resolution, info.height * info.resolution, 1.5,
                info.origin.position.x, info.origin.position.y,
            )
            self.get_logger().debug('PRM klar. tb3_0, tb3_1, tb3_3 og tb3_4 utforsker; tb3_2 er hjelper.')
            self.publish_prm_graph()
        except Exception as e:
            self.get_logger().error(f'Feil ved PRM-init: {e}')

    def clbk_odom(self, msg, r_id):
        frame = (msg.header.frame_id or f'{r_id}/odom').lstrip('/')
        self.odom_frames[r_id] = frame
        try:
            pose = self.to_map(msg.pose.pose, frame, msg.header.stamp)
        except TransformException as ex:
            self.debug_once(('odom_tf', r_id),
                            f'{r_id}: får odometri, men mangler TF fra {frame} til map: {ex}')
            return
        self.reported_waits.discard(('odom_tf', r_id))

        position = (pose.position.x, pose.position.y)
        self.pos[r_id] = position
        if (r_id == self.helper_id and self.helper_route_origin is not None
                and math.dist(position, self.helper_route_origin) >= self.traffic.clearance):
            # Hjelperen har forlatt startplassen sin.
            self.helper_at_spawn = False
            self.helper_route_origin = None
        self.traffic.advance(r_id, position)
        self.record_visited_nodes(position)
        twist = msg.twist.twist
        self.speed[r_id] = (math.hypot(twist.linear.x, twist.linear.y), abs(twist.angular.z))
        self.last_odom[r_id] = self.now()

        rescue = self.rescue
        if rescue and r_id in rescue.targets and not self.rescue_robot_ready(rescue, r_id, self.last_odom[r_id]):
            rescue.still_since = None
            self.pending_reports.pop(FLAME_ID, None)
        if self.manager:
            self.manager.update_robot_position(*position)

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
            if rescue and r_id in rescue.targets and r_id not in rescue.arrived:
                rescue.arrived.add(r_id)
                self.traffic.release(r_id)
                self.get_logger().info(f'{r_id} har kommet frem ved ID 4.')

    def clbk_aruco(self, msg, r_id):
        # Lagre deteksjonen selv om TF for bildet ennå ikke har kommet.
        for marker_id, camera_pose in zip(msg.marker_ids, msg.poses):
            marker_id = int(marker_id)
            if marker_id not in range(TOTAL_TARGETS):
                continue
            if marker_id == FLAME_ID:
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
            # Noen versjoner av aruco_node publiserer et tomt frame_id.
            frame = (msg.header.frame_id or f'{r_id}/camera_rgb_optical_frame').lstrip('/')
            # Behold det opprinnelige tidsstempelet til TF er tilgjengelig.
            # Kontinuerlig overskriving med nyeste bilde kan ellers gi en
            # endeløs rekke ekstrapolasjonsfeil når TF ligger litt etter kamera.
            self.pending_observations.setdefault((r_id, marker_id), (frame, msg.header.stamp, camera_pose))
        self.process_marker_observations()

    # --- Markører og scoring ---

    def process_marker_observations(self):
        for key, (frame, stamp, camera_pose) in list(self.pending_observations.items()):
            r_id, marker_id = key
            if marker_id in self.found_qrs or (marker_id == FLAME_ID and self.rescue is not None):
                del self.pending_observations[key]
                continue
            try:
                pose = self.to_map(camera_pose, frame, stamp)
            except TransformException as ex:
                self.handle_marker_tf_error(key, frame, stamp, ex)
                continue

            if marker_id == FLAME_ID:
                self.reported_waits.discard(('id4_tf_expired', r_id))
                if self.prm is None or self.pos[r_id] is None or self.now() < self.rescue_cooldown_until:
                    # Lagre posisjonen i map før transformasjonen for bildet blir for gammel.
                    self.pending_observations[key] = ('map', stamp, pose)
                    continue

            del self.pending_observations[key]
            self.reported_waits.discard(('marker_tf', r_id, marker_id))
            position = Point(x=float(pose.position.x), y=float(pose.position.y), z=float(pose.position.z))
            if marker_id == FLAME_ID:
                q = pose.orientation
                # Markørens lokale z-akse er normalen til markørflaten.
                normal = (2 * (q.x * q.z + q.w * q.y), 2 * (q.y * q.z - q.w * q.x))
                self.start_rescue(r_id, position, normal)
            else:
                self.register_marker(marker_id, position)

    def handle_marker_tf_error(self, key, frame, stamp, ex):
        r_id, marker_id = key
        self.debug_once(('marker_tf', r_id, marker_id),
                        f'{r_id}: ID {marker_id} er observert, men venter på TF fra {frame} til map: {ex}')
        # Et gammelt bilde kan ligge utenfor TF-bufferens historikk.
        # Gi da plass til en ny deteksjon med et ferskere tidsstempel.
        if self.now() - rclpy.time.Time.from_msg(stamp).nanoseconds <= seconds(3):
            return
        warning_key = ('id4_tf_expired', r_id)
        if marker_id == FLAME_ID and warning_key not in self.reported_waits:
            self.reported_waits.add(warning_key)
            self.get_logger().warning(
                f'{r_id}: ID 4 er lest, men kameraposisjonen kunne ikke '
                f'omregnes fra {frame} til map. Venter på en ny observasjon.')
        del self.pending_observations[key]

    def register_marker(self, marker_id, position):
        if marker_id in range(TOTAL_TARGETS) and marker_id not in self.found_qrs \
                and marker_id not in self.reports_in_flight:
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
        del self.score_futures[marker_id]
        self.reports_in_flight.discard(marker_id)
        try:
            if not future.result().accepted:
                return  # Timeren prøver igjen.
        except Exception as ex:
            if marker_id == FLAME_ID:
                self.release_rescue(f'Feil ved scoring av ID 4: {ex}.')
            else:
                self.get_logger().error(f'Feil ved scoring: {ex}')
            return
        self.pending_reports.pop(marker_id, None)
        self.found_qrs.add(marker_id)
        if marker_id == FLAME_ID and self.rescue is not None:
            # Scoring sjekker nærhet i timeren etter at tjenesten har tatt imot ID 4.
            self.rescue.release_after = self.now() + seconds(1.5)
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

    # --- Redning ved ID 4 ---

    def start_rescue(self, finder_id, position, normal=None):
        now = self.now()
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
        # La normalen peke mot siden kameraet så markøren fra.
        if normal[0] * view[0] + normal[1] * view[1] < 0:
            normal = (-normal[0], -normal[1])

        robots = (finder_id, self.helper_id)
        self.rescue = Rescue(robots=robots, position=position, normal=normal,
                             retry_at=now, warn_after=now + seconds(3))
        for rid in robots:
            self.reported_waits.discard(('id4_wait', rid))
            self.scan_wait_started.pop(rid, None)
            self.held_robots.discard(rid)
            self.hold_pubs[rid].publish(Bool(data=True))
            self.is_exploring[rid] = True
        self.get_logger().debug(f'STOR FLAMME ID 4: {finder_id} stopper. {self.helper_id} sendes fra spawn.')

    def release_rescue(self, reason=None):
        if self.rescue is not None:
            for rid in self.rescue.robots:
                self.traffic.release(rid)
                if not self.robots_stopped:
                    self.hold_pubs[rid].publish(Bool(data=False))
                    # READY bekrefter at handler har behandlet frigivelsen.
                    self.is_exploring[rid] = True
            self.rescue = None
        self.pending_reports.pop(FLAME_ID, None)
        self.reports_in_flight.discard(FLAME_ID)
        self.score_futures.pop(FLAME_ID, None)
        if reason:
            self.get_logger().warning(reason)
            self.rescue_cooldown_until = self.now() + seconds(20)

    def in_bounds(self, gx, gy):
        return 0 <= gx < self.prm.width and 0 <= gy < self.prm.height

    def flame_visible(self, point, rescue):
        """Sjekk om markøren er synlig fra point i råkartet."""
        flame = rescue.flame
        normal = rescue.normal
        # Samme side av markørflaten som kameraet som observerte den.
        if (point[0] - flame[0]) * normal[0] + (point[1] - flame[1]) * normal[1] <= 0.10:
            return False
        # Bruk råkartet for sikt, ikke robotens oppblåste sikkerhetsområde.
        gx, gy = self.prm.world_to_grid(*flame)
        if not self.in_bounds(gx, gy):
            return False
        marker_in_wall = self.prm.grid[gy, gx] >= self.prm.obstacle_threshold
        distance = math.dist(point, flame)
        steps = max(1, math.ceil(distance / (self.prm.resolution / 3)))
        hit_wall = False
        for i in range(steps + 1):
            t = i / steps
            gx, gy = self.prm.world_to_grid(point[0] + t * (flame[0] - point[0]),
                                            point[1] + t * (flame[1] - point[1]))
            if not self.in_bounds(gx, gy):
                return False
            value = self.prm.grid[gy, gx]
            if value < 0:
                return False
            if value >= self.prm.obstacle_threshold:
                # Godta bare veggceller i den siste sammenhengende biten ved markøren.
                if not marker_in_wall or distance * (1 - t) > 0.25:
                    return False
                hit_wall = True
            elif hit_wall:
                return False
        return True

    def find_rescue_routes(self, rescue, ready_robots):
        flame = rescue.flame
        outer = self.rescue_radius - 0.15
        clearance = self.traffic.clearance

        # Kandidatstopp: PRM-noder, robotenes posisjoner og sirkler rundt flammen.
        candidate_points = [*self.prm.nodes, *(self.pos[r] for r in ready_robots)]
        candidates = [p for p in candidate_points if 0.40 <= math.dist(p, flame) <= outer]
        for radius in (0.45, 0.65, 1.0, outer):
            if not 0.40 <= radius <= outer:
                continue
            for i in range(24):
                angle = i * math.tau / 24
                point = (flame[0] + radius * math.cos(angle), flame[1] + radius * math.sin(angle))
                if self.prm.is_free(*point):
                    candidates.append(point)
        candidates = [p for p in candidates if self.flame_visible(p, rescue)
                      and all(math.dist(p, t) >= clearance for t in rescue.targets.values())]

        # options[robot] = [(langt_unna, lengde, rute), ...]. Lavest er best.
        options = {}
        for robot_id in ready_robots:
            # ID 4 bruker kartet og kjørelengden. Lidar håndterer hindringer under kjøring.
            routes = self.traffic.shortest_routes(
                self.prm, robot_id, self.pos[robot_id], candidates, {}, ignore_robots=self.robot_ids,
                allow_start_escape=True,
            )
            options[robot_id] = [
                (math.dist(route[-1], flame) > 0.80,
                 sum(math.dist(a, b) for a, b in zip(route, route[1:])), route)
                for route in routes
            ]

        if len(ready_robots) == 2:
            # Velg det beste paret av stopp som ikke står for nær hverandre.
            first, second = ready_robots
            pair = min(
                ((far1 + far2, cost1 + cost2, route1, route2)
                 for far1, cost1, route1 in options[first]
                 for far2, cost2, route2 in options[second]
                 if math.dist(route1[-1], route2[-1]) >= clearance),
                key=lambda option: option[:2], default=None,
            )
            if pair is not None:
                return {first: pair[2], second: pair[3]}

        # Planlegg hver robot for seg. Uten PRM-rute brukes kartcellene direkte.
        planned = {}
        stops = list(rescue.targets.values())
        for robot_id in ready_robots:
            best = min(
                (option for option in options[robot_id]
                 if all(math.dist(option[2][-1], stop) >= clearance for stop in stops)),
                key=lambda option: option[:2], default=None,
            )
            route = best[2] if best else self.grid_rescue_route(self.pos[robot_id], flame, rescue.normal, stops)
            if route:
                planned[robot_id] = route
                stops.append(route[-1])
        return planned

    def grid_rescue_route(self, start, flame, normal, stops):
        outer = self.rescue_radius - 0.15
        radius = math.ceil(outer / self.prm.resolution)
        fx, fy = self.prm.world_to_grid(*flame)
        goals, close, preferred = set(), set(), set()
        for y in range(max(0, fy - radius), min(self.prm.height, fy + radius + 1)):
            for x in range(max(0, fx - radius), min(self.prm.width, fx + radius + 1)):
                if not self.prm.navigable[y, x]:
                    continue
                point = self.prm.grid_to_world(x, y)
                distance = math.dist(point, flame)
                if not 0.40 <= distance <= outer or any(
                        math.dist(point, stop) < self.traffic.clearance for stop in stops):
                    continue
                goals.add((x, y))
                if distance <= 0.80:
                    close.add((x, y))
                    side = (point[0] - flame[0]) * normal[0] + (point[1] - flame[1]) * normal[1]
                    if side > 0.10:
                        preferred.add((x, y))
        return (self.prm.shortest_grid_route(start, preferred)
                or self.prm.shortest_grid_route(start, close)
                or self.prm.shortest_grid_route(start, goals))

    def rescue_robot_ready(self, rescue, robot_id, now):
        """Står roboten stille ved stoppunktet sitt, nær nok flammen?"""
        target = rescue.targets.get(robot_id)
        position = self.pos[robot_id]
        stamp = self.last_odom[robot_id]
        linear, angular = self.speed[robot_id]
        return (target is not None and robot_id in rescue.arrived and stamp is not None
                and 0 <= now - stamp < SEC and position is not None
                and math.dist(position, target) <= 0.15
                and math.dist(position, rescue.flame) <= self.rescue_radius
                and linear < 0.02 and angular < 0.03)

    def process_rescue(self):
        rescue = self.rescue
        if rescue is None:
            return
        now = self.now()
        if rescue.release_after is not None:
            if now >= rescue.release_after:
                self.release_rescue()
            return
        if rescue.report_wait_since is not None and now - rescue.report_wait_since >= seconds(10):
            self.release_rescue()
            self.rescue_cooldown_until = now + seconds(20)
            return

        waiting = [r for r in rescue.robots if r not in rescue.targets]
        if waiting:
            self.send_rescue_routes(rescue, waiting, now)
            return

        if not all(self.rescue_robot_ready(rescue, r, now) for r in rescue.robots):
            rescue.still_since = None
            self.pending_reports.pop(FLAME_ID, None)
            return
        if rescue.still_since is None:
            rescue.still_since = now
            self.get_logger().debug(
                f'Begge roboter står ved ID 4. Venter {self.rescue_hold_seconds:.1f} sekunder.')
        if now - rescue.still_since >= seconds(self.rescue_hold_seconds):
            if rescue.report_wait_since is None:
                rescue.report_wait_since = now
            self.register_marker(FLAME_ID, rescue.position)

    def send_rescue_routes(self, rescue, waiting, now):
        for rid in waiting:
            if rid not in self.held_robots:
                self.hold_pubs[rid].publish(Bool(data=True))
        ready = [r for r in waiting if r in self.held_robots and self.pos[r] is not None
                 and self.rescue_path_pubs[r].get_subscription_count() > 0]

        if ready and now >= rescue.retry_at:
            rescue.retry_at = now + seconds(3)
            for robot_id, points in self.find_rescue_routes(rescue, ready).items():
                path = self.make_path_msg(robot_id, points)
                if path is None:
                    continue
                rescue.targets[robot_id] = points[-1]
                if robot_id == self.helper_id and self.helper_at_spawn:
                    self.helper_route_origin = self.pos[robot_id]
                self.traffic.reserve(robot_id, points)
                self.rescue_path_pubs[robot_id].publish(path)
                self.get_logger().info(f'{robot_id}: redningsrute til ID 4 sendt.')

        if now >= rescue.warn_after:
            for rid in waiting:
                key = ('id4_wait', rid)
                if rid in rescue.targets or key in self.reported_waits:
                    continue
                reason = ('ingen kjørbar rute eller TF for veipunktene' if rid in ready
                          else 'venter på kartposisjon eller bekreftelse fra robot_handler')
                self.get_logger().warning(f'{rid}: venter på redningsrute til ID 4 ({reason}).')
                self.reported_waits.add(key)

    # --- Utforskning ---

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
        if self.robots_stopped or self.prm is None or self.manager is None:
            return
        reserved = self.rescue.robots if self.rescue else ()
        for r_id in self.robot_ids:
            if r_id != self.helper_id and r_id not in reserved:
                self.explore(r_id)

    def explore(self, r_id):
        if self.pos[r_id] is None:
            if r_id in ('tb3_0', 'tb3_1'):
                self.debug_once(('position', r_id),
                                f'{r_id}: venter på robotposisjon i map. Sjekk odometri og TF.')
            return
        self.reported_waits.discard(('position', r_id))
        if self.is_exploring[r_id]:
            return
        if self.path_pubs[r_id].get_subscription_count() == 0:
            self.debug_once(('subscriber', r_id),
                            f'{r_id}: venter på robot_handler som abonnent på /{r_id}/path.')
            return
        self.reported_waits.discard(('subscriber', r_id))
        self.assign_new_target(r_id)
        if not self.is_exploring[r_id]:
            self.request_left_scan(r_id)

    def request_left_scan(self, robot_id):
        """Be roboten skanne rundt seg hvis den har ventet på rute i 3 sekunder."""
        now = self.now()
        started = self.scan_wait_started.setdefault(robot_id, now)
        if started is None or now - started < seconds(3):
            return
        publisher = self.scan_left_pubs[robot_id]
        if publisher.get_subscription_count() > 0:
            publisher.publish(Empty())
            # Én skann per venteperiode. Neste rute nullstiller ventingen.
            self.scan_wait_started[robot_id] = None

    def assign_new_target(self, robot_id):
        start_pos = self.pos[robot_id]
        if not self.prm.is_free(*start_pos):
            self.debug_once(('blocked_start', robot_id),
                            f'{robot_id}: posisjon {start_pos} er utenfor ledig PRM-område. '
                            'Sjekk map/odom, spawnposisjon og robot_radius.')
            return
        self.reported_waits.discard(('blocked_start', robot_id))

        target = self.manager.get_next_target(robot_id, self.prm.nodes, *start_pos)
        if not target:
            self.debug_once(('target', robot_id), f'{robot_id}: ingen mål tilgjengelig i PRM.')
            return
        if self.send_path(robot_id, target):
            self.is_exploring[robot_id] = True
            self.scan_wait_started.pop(robot_id, None)
            self.reported_waits.discard(('route', robot_id))
        else:
            self.debug_once(('route', robot_id),
                            f'{robot_id}: fant ikke sendbar rute fra {start_pos} til {target}. '
                            'Sjekk PRM-forbindelser og TF.')

    def send_path(self, robot_id, target):
        points = self.traffic.plan(
            self.prm, AStar, robot_id, self.pos[robot_id], target,
            self.planning_positions(robot_id), zone_side=EXPLORATION_SIDES.get(robot_id),
        )
        if not points:
            return False
        # Første punkt er robotens nåværende posisjon, ikke et stopp. Send maks 20 veipunkter.
        sent_points = points[:21]
        path_msg = self.make_path_msg(robot_id, sent_points[1:])
        if path_msg is None or not path_msg.poses:
            return False
        self.traffic.reserve(robot_id, sent_points)
        self.path_pubs[robot_id].publish(path_msg)
        return True

    def planning_positions(self, robot_id):
        if robot_id == self.helper_id or not self.helper_at_spawn:
            return self.pos
        # Sperr hjelperens startplass, også før odometri og TF er tilgjengelig.
        return {**self.pos, 'helper_spawn': self.helper_spawn}

    def make_path_msg(self, robot_id, points):
        """Lag en Path i robotens odometriramme fra punkter i map.

        Robot_handler sammenligner med rå odometri, mens planleggingen bruker map.
        """
        frame = self.odom_frames.get(robot_id)
        if frame is None:
            return None
        try:
            transform = None
            if frame != 'map':
                transform = self.tf_buffer.lookup_transform(frame, 'map', rclpy.time.Time())
        except TransformException as ex:
            self.debug_once(('path_tf', robot_id),
                            f'{robot_id}: kan ikke omregne ruten fra map til {frame}: {ex}')
            return None

        message = Path()
        message.header.frame_id = frame
        message.header.stamp = self.get_clock().now().to_msg()
        for x, y in points:
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            if transform is not None:
                pose.pose = tf2_geometry_msgs.do_transform_pose(pose.pose, transform)
            message.poses.append(pose)
        return message

    # --- Visualisering ---

    def record_visited_nodes(self, position):
        if self.prm is None:
            return
        # Grønt betyr faktisk besøkt, ikke valgt mål eller planlagt rute.
        for node_id, point in enumerate(self.prm.nodes):
            if (node_id not in self.visited_node_ids and math.dist(position, point) <= 0.10
                    and self.prm.line_is_free(position, point)):
                self.visited_node_ids.add(node_id)
                self.visited_points.append(point)
                self.visited_dirty = True

    def publish_active_routes(self):
        markers = MarkerArray()
        visited = set(self.visited_points)
        yellow = (1.0, 1.0, 0.0, 1.0)
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
            line.color.r, line.color.g, line.color.b, line.color.a = yellow
            line.points = [Point(x=float(x), y=float(y), z=0.08) for x, y in points]
            markers.markers.append(line)

            # Første punkt er robotens posisjon; resten er gjenstående veipunkter.
            remaining = [p for p in points[1:] if p not in visited]
            nodes = sphere_list('reserved_nodes', marker_id, 0.18, yellow, remaining, 0.08)
            nodes.action = Marker.ADD if points else Marker.DELETE
            markers.markers.append(nodes)
        self.route_marker_pub.publish(markers)

    def publish_visited_nodes(self):
        marker = sphere_list('visited_nodes', 10, 0.2, (0.0, 1.0, 0.0, 1.0), self.visited_points, 0.1)
        self.visited_pub.publish(MarkerArray(markers=[marker]))

    def publish_prm_graph(self):
        """Tegner PRM-nodene som røde prikker i RViz."""
        if self.prm is None:
            return
        marker = sphere_list('prm_nodes', 0, 0.15, (1.0, 0.0, 0.0, 0.8), self.prm.nodes, 0.05)
        self.prm_marker_pub.publish(MarkerArray(markers=[marker]))


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
