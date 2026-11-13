import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Bool, Empty

ROBOT_IDS = ['tb3_0', 'tb3_1', 'tb3_2', 'tb3_3', 'tb3_4']
HELPER_ID = 'tb3_2'  # Hjelperen skanner ikke mens den venter.

WAYPOINT_TOLERANCE = 0.1   # m
LINEAR_SPEED = 0.3         # m/s
TURN_SPEED = 0.5           # rad/s når roboten snur på stedet
SCAN_SPEED = 0.25          # rad/s under venteskann
MAX_HEADING_ERROR = 0.15   # rad før roboten snur på stedet
MIN_FRONT_DISTANCE = 0.25  # m, stopp hvis noe er rett foran
SCAN_TIMEOUT_NS = 35_000_000_000


def quaternion_to_yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap_angle(angle):
    return (angle + math.pi) % math.tau - math.pi


def path_points(msg):
    return [(p.pose.position.x, p.pose.position.y) for p in msg.poses]


class RobotState:
    def __init__(self, name):
        self.name = name
        self.lidar = 100.0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.path = []
        self.state = 'IDLE'      # IDLE, NAVIGATE eller HOLD
        self.rescue = False      # True når roboten kjører/står ved ID 4
        self.odom_received = False
        self.scan_started = None  # Tidspunkt for start av venteskann
        self.scan_yaw = None
        self.scan_angle = 0.0


class RobotHandlerClass(Node):
    def __init__(self):
        super().__init__('RobotHandlerQuadNode')

        self.robots = [RobotState(name) for name in ROBOT_IDS]
        self.pubs_cmd_vel = []
        self.pubs_status = []
        self.mission_stopped = False

        for i, name in enumerate(ROBOT_IDS):
            def subscribe(msg_type, topic, callback, qos=10, idx=i):
                self.create_subscription(
                    msg_type, f'/{name}/{topic}', lambda msg: callback(msg, idx), qos)

            subscribe(LaserScan, 'scan', self.clbk_lidar, qos_profile_sensor_data)
            subscribe(Odometry, 'odom', self.clbk_odom, qos_profile_sensor_data)
            subscribe(Path, 'path', self.clbk_path)
            subscribe(Empty, 'scan_left', self.clbk_scan_left)
            subscribe(Bool, 'hold', self.clbk_hold)              # Stopp eller frigi roboten
            subscribe(Path, 'rescue_path', self.clbk_rescue_path)  # Prioritert rute til ID 4

            self.pubs_cmd_vel.append(self.create_publisher(Twist, f'/{name}/cmd_vel', 10))
            self.pubs_status.append(self.create_publisher(String, f'/{name}/status', 10))

        stop_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, '/mission_stop', self.clbk_mission_stop, stop_qos)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('Robot_handler klar for fem roboter. Venter på odometri og ruter.')

    def publish_cmd(self, r_id, twist=None):
        self.pubs_cmd_vel[r_id].publish(twist or Twist())

    def publish_status(self, r_id, status):
        self.pubs_status[r_id].publish(String(data=status))

    # --- Callbacks ---

    def clbk_lidar(self, msg, r_id):
        if msg.ranges:
            front = msg.ranges[0]
            self.robots[r_id].lidar = front if front != float('inf') else 100.0

    def clbk_odom(self, msg, r_id):
        r = self.robots[r_id]
        if not r.odom_received:
            self.get_logger().info(f'{r.name}: mottar odometri.')
        r.odom_received = True
        r.x, r.y = msg.pose.pose.position.x, msg.pose.pose.position.y
        r.yaw = quaternion_to_yaw(msg.pose.pose.orientation)

    def clbk_path(self, msg, r_id):
        r = self.robots[r_id]
        if self.mission_stopped or r.rescue or r.state != 'IDLE' or not msg.poses:
            return
        if r.scan_started is not None:
            self.publish_cmd(r_id)  # Avbryt venteskann
        r.path = path_points(msg)
        r.state = 'NAVIGATE'
        r.scan_started = None
        self.get_logger().info(f'{r.name} mottok rute med {len(msg.poses)} punkter. Kjører direkte!')

    def clbk_mission_stop(self, msg):
        if not msg.data or self.mission_stopped:
            return
        self.mission_stopped = True
        for r_id, r in enumerate(self.robots):
            r.path = []
            r.state = 'HOLD'
            self.publish_cmd(r_id)

    def clbk_hold(self, msg, r_id):
        if self.mission_stopped:
            return
        r = self.robots[r_id]
        # Avbryt gjeldende rute og stopp roboten
        r.path = []
        r.rescue = msg.data
        r.scan_started = None
        r.state = 'HOLD' if msg.data else 'IDLE'
        self.publish_cmd(r_id)
        self.publish_status(r_id, 'HOLDING' if msg.data else 'READY')

    def clbk_rescue_path(self, msg, r_id):
        if self.mission_stopped or not msg.poses:
            return
        r = self.robots[r_id]
        # Redningsruten erstatter den vanlige ruten
        r.rescue = True
        r.path = path_points(msg)
        r.state = 'NAVIGATE'
        r.scan_started = None
        self.get_logger().info(f'{r.name} mottok redningsrute til stor flamme.')

    def clbk_scan_left(self, msg, r_id):
        r = self.robots[r_id]
        if (self.mission_stopped or r.state != 'IDLE' or not r.odom_received
                or r.scan_started is not None or r.name == HELPER_ID):
            return
        r.scan_started = self.get_clock().now().nanoseconds
        r.scan_yaw = r.yaw
        r.scan_angle = 0.0

    # --- Styring ---

    def scan_while_waiting(self, r_id):
        """Roter én hel runde (eller til tidsavbrudd) mens roboten venter på rute."""
        r = self.robots[r_id]
        if r.scan_started is None or r.scan_angle >= math.tau:
            return

        r.scan_angle += wrap_angle(r.yaw - r.scan_yaw)
        r.scan_yaw = r.yaw
        if self.get_clock().now().nanoseconds - r.scan_started >= SCAN_TIMEOUT_NS:
            r.scan_angle = math.tau

        twist = Twist()
        if r.scan_angle < math.tau:
            twist.angular.z = SCAN_SPEED
        self.publish_cmd(r_id, twist)

    def arrive_at_waypoint(self, r_id):
        r = self.robots[r_id]
        r.path.pop(0)
        self.publish_cmd(r_id)
        if r.path:
            return

        if r.rescue:
            # Vent ved flammen til leader frigir roboten
            r.state = 'HOLD'
            self.get_logger().info(f'{r.name} ankom redningsmålet. Venter.')
            self.publish_status(r_id, 'RESCUE_READY')
        else:
            r.state = 'IDLE'
            self.get_logger().info(f'{r.name} ankom målet. Ber om ny rute.')
            self.publish_status(r_id, 'READY')

    def drive_towards_waypoint(self, r_id):
        r = self.robots[r_id]
        tx, ty = r.path[0]
        if math.dist((r.x, r.y), (tx, ty)) < WAYPOINT_TOLERANCE:
            self.arrive_at_waypoint(r_id)
            return

        angle_diff = wrap_angle(math.atan2(ty - r.y, tx - r.x) - r.yaw)
        twist = Twist()
        if abs(angle_diff) > MAX_HEADING_ERROR:
            # Snu på stedet mot veipunktet
            twist.angular.z = TURN_SPEED if angle_diff > 0 else -TURN_SPEED
        elif r.lidar >= MIN_FRONT_DISTANCE:
            twist.angular.z = 0.8 * angle_diff
            twist.linear.x = LINEAR_SPEED
        self.publish_cmd(r_id, twist)

    def process_robot(self, r_id):
        r = self.robots[r_id]
        if r.state == 'HOLD':
            # Bli stående til leader sender redningsrute eller frigir roboten
            self.publish_cmd(r_id)
        elif r.state == 'IDLE':
            self.scan_while_waiting(r_id)
        elif r.state == 'NAVIGATE':
            if not r.odom_received:
                self.publish_cmd(r_id)
            elif r.path:
                self.drive_towards_waypoint(r_id)

    def timer_callback(self):
        for r_id in range(len(self.robots)):
            self.process_robot(r_id)


def main(args=None):
    rclpy.init(args=args)
    node = RobotHandlerClass()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
