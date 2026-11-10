import rclpy
import math
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String, Bool, Empty
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data

class RobotHandlerClass(Node):
    def __init__(self):
        super().__init__('RobotHandlerQuadNode')

        # Liste over alle 5 robotene
        self.robot_ids = ["tb3_0", "tb3_1", "tb3_2", "tb3_3", "tb3_4"]

        self.pubs_cmd_vel = []
        self.pubs_status = []
        self.robots = []
        self.mission_stopped = False

        for i, r_id in enumerate(self.robot_ids):
            # Dynamisk oppretting av abonnementer for hver robot
            self.create_subscription(LaserScan, f'/{r_id}/scan', lambda msg, idx=i: self.clbk_lidar(msg, idx), qos_profile_sensor_data)
            self.create_subscription(Odometry, f'/{r_id}/odom', lambda msg, idx=i: self.clbk_odom(msg, idx), qos_profile_sensor_data)
            self.create_subscription(Path, f'/{r_id}/path', lambda msg, idx=i: self.clbk_path(msg, idx), 10)
            self.create_subscription(
                Empty, f'/{r_id}/scan_left',
                lambda msg, idx=i: self.clbk_scan_left(msg, idx), 10,
            )

            # Kommando for å stoppe eller frigi roboten
            self.create_subscription(
                Bool, f'/{r_id}/hold',
                lambda msg, idx=i: self.clbk_hold(msg, idx), 10
            )

            # Prioritert rute til stor flamme
            self.create_subscription(
                Path, f'/{r_id}/rescue_path',
                lambda msg, idx=i: self.clbk_rescue_path(msg, idx), 10
            )

            # Opprett publishere
            self.pubs_cmd_vel.append(self.create_publisher(Twist, f'/{r_id}/cmd_vel', 10))
            self.pubs_status.append(self.create_publisher(String, f'/{r_id}/status', 10))

            self.robots.append(self.init_robot_state())

        stop_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, '/mission_stop', self.clbk_mission_stop, stop_qos)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('Robot_handler klar for fem roboter. Venter på odometri og ruter.')

    def init_robot_state(self):
        return {
            'lidar': 100.0,
            'x': 0.0,
            'y': 0.0,
            'yaw': 0.0,
            'path': [],
            'state': 'IDLE',
            'rescue': False,
            'odom_received': False,
            'scan_started': None,
            'scan_yaw': None,
            'scan_angle': 0.0,
        }

    def clbk_lidar(self, msg, r_id):
        if len(msg.ranges) > 0:
            self.robots[r_id]['lidar'] = msg.ranges[0] if msg.ranges[0] != float('inf') else 100.0

    def clbk_odom(self, msg, r_id):
        if not self.robots[r_id]['odom_received']:
            self.get_logger().info(f'{self.robot_ids[r_id]}: mottar odometri.')
        self.robots[r_id]['odom_received'] = True
        self.robots[r_id]['x'], self.robots[r_id]['y'] = msg.pose.pose.position.x, msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robots[r_id]['yaw'] = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def clbk_path(self, msg, r_id):
        if (
            not self.mission_stopped
            and not self.robots[r_id]['rescue']
            and self.robots[r_id]['state'] == 'IDLE'
            and len(msg.poses) > 0
        ):
            r = self.robots[r_id]
            if r['scan_started'] is not None:
                self.pubs_cmd_vel[r_id].publish(Twist())
            r['path'] = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
            r['state'] = 'NAVIGATE'
            r['scan_started'] = None
            robot_name = self.robot_ids[r_id]
            self.get_logger().info(f'{robot_name} mottok rute med {len(msg.poses)} punkter. Kjører direkte!')

    def clbk_mission_stop(self, msg):
        if not msg.data or self.mission_stopped:
            return
        self.mission_stopped = True
        for r, publisher in zip(self.robots, self.pubs_cmd_vel):
            r['path'] = []
            r['state'] = 'HOLD'
            publisher.publish(Twist())

    def clbk_hold(self, msg, r_id):
        if self.mission_stopped:
            return
        r = self.robots[r_id]

        # Avbryt gjeldende rute og stopp roboten
        r['path'] = []
        r['rescue'] = msg.data
        r['scan_started'] = None

        status_msg = String()

        if msg.data:
            r['state'] = 'HOLD'
            status_msg.data = "HOLDING"
        else:
            r['state'] = 'IDLE'
            status_msg.data = "READY"

        self.pubs_cmd_vel[r_id].publish(Twist())
        self.pubs_status[r_id].publish(status_msg)

    def clbk_rescue_path(self, msg, r_id):
        if self.mission_stopped or len(msg.poses) == 0:
            return

        r = self.robots[r_id]

        # Redningsruten erstatter den vanlige ruten
        r['rescue'] = True
        r['path'] = [
            (p.pose.position.x, p.pose.position.y)
            for p in msg.poses
        ]
        r['state'] = 'NAVIGATE'
        r['scan_started'] = None

        robot_name = self.robot_ids[r_id]
        self.get_logger().info(
            f'{robot_name} mottok redningsrute til stor flamme.'
        )

    def clbk_scan_left(self, msg, r_id):
        r = self.robots[r_id]
        if (self.mission_stopped or r['state'] != 'IDLE' or not r['odom_received']
                or r['scan_started'] is not None or self.robot_ids[r_id] == 'tb3_2'):
            return
        r['scan_started'] = self.get_clock().now().nanoseconds
        r['scan_yaw'] = r['yaw']
        r['scan_angle'] = 0.0

    def scan_while_waiting(self, r_id, twist):
        r = self.robots[r_id]
        if r['scan_started'] is None or r['scan_angle'] >= math.tau:
            return

        yaw = r['yaw']
        r['scan_angle'] += (yaw - r['scan_yaw'] + math.pi) % math.tau - math.pi
        r['scan_yaw'] = yaw
        elapsed = self.get_clock().now().nanoseconds - r['scan_started']
        if elapsed >= 35_000_000_000:
            r['scan_angle'] = math.tau
        if r['scan_angle'] < math.tau:
            twist.angular.z = 0.25
        self.pubs_cmd_vel[r_id].publish(twist)

    def process_robot(self, r_id):
        r = self.robots[r_id]
        twist = Twist()
        robot_name = self.robot_ids[r_id]

        # Bli stående til leader sender redningsrute eller frigir roboten
        if r['state'] == 'HOLD':
            self.pubs_cmd_vel[r_id].publish(twist)
            return

        if r['state'] == 'IDLE':
            self.scan_while_waiting(r_id, twist)
            return

        if r['state'] == 'NAVIGATE' and not r['odom_received']:
            self.pubs_cmd_vel[r_id].publish(twist)
            return

        if r['state'] == 'NAVIGATE' and r['path']:
            tx, ty = r['path'][0]
            dx, dy = tx - r['x'], ty - r['y']

            # Sjekk om vi er fremme ved veipunktet (innenfor 10 cm)
            if math.dist((r['x'], r['y']), (tx, ty)) < 0.1:
                r['path'].pop(0)  # Fjern punktet vi nådde

                if not r['path']:
                    status_msg = String()

                    if r['rescue']:
                        # Vent ved flammen til leader frigir roboten
                        r['state'] = 'HOLD'
                        status_msg.data = "RESCUE_READY"
                        self.get_logger().info(
                            f'{robot_name} ankom redningsmålet. Venter.'
                        )
                    else:
                        # Ferdig med vanlig rute
                        r['state'] = 'IDLE'
                        status_msg.data = "READY"
                        self.get_logger().info(
                            f'{robot_name} ankom målet. Ber om ny rute.'
                        )

                self.pubs_cmd_vel[r_id].publish(twist)
                if not r['path']:
                    self.pubs_status[r_id].publish(status_msg)
                return

            # Bekreft ankomst selv om det er en hindring foran roboten.
            if r['lidar'] < 0.25:
                self.pubs_cmd_vel[r_id].publish(twist)
                return

            # Beregn vinkelavvik mot målet
            angle_diff = (math.atan2(dy, dx) - r['yaw'] + math.pi) % (2 * math.pi) - math.pi

            # Jevn og stabil regulering mot målet
            if abs(angle_diff) > 0.15:
                twist.angular.z = 0.5 if angle_diff > 0 else -0.5
                twist.linear.x = 0.0
            else:
                twist.angular.z = 0.8 * angle_diff
                twist.linear.x = 0.22

            self.pubs_cmd_vel[r_id].publish(twist)

    def timer_callback(self):
        # Behandle alle 5 robotene i løkka hver 100ms
        for i in range(len(self.robot_ids)):
            self.process_robot(i)

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
