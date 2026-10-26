import rclpy
import math
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.qos import qos_profile_sensor_data

class RobotHandlerClass(Node):
    def __init__(self):
        super().__init__('RobotHandlerDualNode')

        self.create_subscription(LaserScan, '/tb3_0/scan', lambda msg: self.clbk_lidar(msg, 0), qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/tb3_1/scan', lambda msg: self.clbk_lidar(msg, 1), qos_profile_sensor_data)
        self.create_subscription(Odometry, '/tb3_0/odom', lambda msg: self.clbk_odom(msg, 0), 10)
        self.create_subscription(Odometry, '/tb3_1/odom', lambda msg: self.clbk_odom(msg, 1), 10)
        self.create_subscription(Path, '/tb3_0/path', lambda msg: self.clbk_path(msg, 0), 10)
        self.create_subscription(Path, '/tb3_1/path', lambda msg: self.clbk_path(msg, 1), 10)

        self.pubs_cmd_vel = [
            self.create_publisher(Twist, '/tb3_0/cmd_vel', 10),
            self.create_publisher(Twist, '/tb3_1/cmd_vel', 10)
        ]

        self.robots = [self.init_robot_state(), self.init_robot_state()]
        self.timer = self.create_timer(0.1, self.timer_callback)

    def init_robot_state(self):
        return {
            'lidar': 100.0, 'x': 0.0, 'y': 0.0, 'yaw': 0.0,
            'path': [], 'state': 'IDLE', 'scan_start': 0.0, 'scanned': 0.0
        }

    def clbk_lidar(self, msg, r_id):
        if len(msg.ranges) > 0:
            self.robots[r_id]['lidar'] = msg.ranges[0] if msg.ranges[0] != float('inf') else 100.0

    def clbk_odom(self, msg, r_id):
        self.robots[r_id]['x'] = msg.pose.pose.position.x
        self.robots[r_id]['y'] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robots[r_id]['yaw'] = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def clbk_path(self, msg, r_id):
        if self.robots[r_id]['state'] == 'IDLE':
            for pose in msg.poses:
                self.robots[r_id]['path'].append((pose.pose.position.x, pose.pose.position.y))
            self.robots[r_id]['state'] = 'NAVIGATE'
            robot_name = "tb3_0" if r_id == 0 else "tb3_1"
            self.get_logger().info(f'{robot_name} mottok rute med {len(msg.poses)} punkter. Kjører!')

    def process_robot(self, r_id):
        r = self.robots[r_id]
        twist = Twist()
        robot_name = "tb3_0" if r_id == 0 else "tb3_1"

        if r['lidar'] < 0.25 and r['state'] == 'NAVIGATE':
            self.pubs_cmd_vel[r_id].publish(twist)
            return

        if r['state'] == 'SCAN':
            angle_diff = abs(r['yaw'] - r['scan_start'])
            if angle_diff > math.pi:
                angle_diff = (2 * math.pi) - angle_diff
            
            r['scanned'] += angle_diff
            r['scan_start'] = r['yaw']

            if r['scanned'] >= (2 * math.pi - 0.2):
                r['path'].pop(0)
                r['state'] = 'NAVIGATE'
                self.get_logger().info(f'{robot_name} ferdig med å skanne. Fortsetter.')
            else:
                twist.angular.z = 0.6
            self.pubs_cmd_vel[r_id].publish(twist)
            return

        if r['state'] == 'NAVIGATE' and r['path']:
            tx, ty = r['path'][0]
            dx, dy = tx - r['x'], ty - r['y']
            
            if math.dist((r['x'], r['y']), (tx, ty)) < 0.25:
                r['state'] = 'SCAN'
                r['scan_start'] = r['yaw']
                r['scanned'] = 0.0
                self.get_logger().info(f'{robot_name} ankom veipunkt. Starter 360-graders skanning.')
                self.pubs_cmd_vel[r_id].publish(twist)
                return

            angle_diff = (math.atan2(dy, dx) - r['yaw'] + math.pi) % (2 * math.pi) - math.pi

            if abs(angle_diff) > 0.15:
                twist.angular.z = 0.4 if angle_diff > 0 else -0.4
            else:
                twist.linear.x = 0.2

            self.pubs_cmd_vel[r_id].publish(twist)

    def timer_callback(self):
        self.process_robot(0)
        self.process_robot(1)

def main(args=None):
    rclpy.init(args=args)
    node = RobotHandlerClass()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()