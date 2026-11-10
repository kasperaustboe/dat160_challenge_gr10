import hashlib
import json
import math
from pathlib import Path as FilePath

import numpy as np
import rclpy

from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker
from rclpy.duration import Duration


class WallCoverage(Node):
    def __init__(self):
        super().__init__("wall_coverage")

        # Kontroller image_topic mot deres faktiske ROS-topics.
        defaults = {
            "map_topic": "/map",
            "image_topic": "/tb3_0/camera/rgb/image_raw",

            # Tom streng: bruk frame_id fra kamerabildet.
            # Denne rammen må være kameraets OPTICAL frame.
            "camera_frame": "",

            "horizontal_fov_deg": 60.0,
            "min_range": 0.25,
            "max_range": 2.0,
            "max_incidence_deg": 65.0,
            "obstacle_threshold": 50,
            "update_period": 0.25,
            "display_height": 0.05,
            "save_path": "wall_coverage.json",
        }

        for name, value in defaults.items():
            self.declare_parameter(name, value)

        def parameter(name):
            return self.get_parameter(name).value

        self.camera_frame = parameter("camera_frame")
        self.half_fov = math.radians(
            parameter("horizontal_fov_deg") / 2.0
        )
        self.min_range = float(parameter("min_range"))
        self.max_range = float(parameter("max_range"))
        self.min_incidence_cos = math.cos(
            math.radians(parameter("max_incidence_deg"))
        )
        self.threshold = int(parameter("obstacle_threshold"))
        self.update_period = float(parameter("update_period"))
        self.display_height = float(parameter("display_height"))
        self.save_path = parameter("save_path")

        if not 0.0 < self.half_fov < math.pi / 2:
            raise ValueError("horizontal_fov_deg må være mellom 0 og 180")
        if not 0 <= self.min_range < self.max_range:
            raise ValueError("Krever 0 <= min_range < max_range")
        if self.update_period <= 0:
            raise ValueError("update_period må være positiv")

        self.grid = None
        self.walls = []
        self.seen = set()
        self.visible = set()
        self.last_image_stamp = None
        self.last_success_time = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pending_image = None
        self.image_timer = self.create_timer(
            0.05, self.process_pending_image
        )

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.marker_pub = self.create_publisher(
            Marker, "/wall_coverage", latched_qos
        )

        self.create_subscription(
            OccupancyGrid,
            parameter("map_topic"),
            self.on_map,
            latched_qos,
        )

        self.create_subscription(
            Image,
            parameter("image_topic"),
            self.on_image,
            qos_profile_sensor_data,
        )

        self.create_service(
            Trigger, "/wall_coverage/save", self.save_coverage
        )

        self.create_timer(0.5, self.publish_coverage)

        self.get_logger().info(
            "Venter på kart, kamerabilder og TF. "
            "RViz-topic: /wall_coverage"
        )

    def on_map(self, msg):
        # Oppgaven bruker et fast kart.
        if self.grid is not None:
            return

        width = msg.info.width
        height = msg.info.height

        if (
            width == 0
            or height == 0
            or msg.info.resolution <= 0
            or len(msg.data) != width * height
            or not msg.header.frame_id
        ):
            self.get_logger().error("Ugyldig OccupancyGrid")
            return

        self.grid = np.asarray(
            msg.data, dtype=np.int16
        ).reshape(height, width)

        self.height, self.width = self.grid.shape
        self.resolution = float(msg.info.resolution)
        self.map_frame = msg.header.frame_id

        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y

        q = msg.info.origin.orientation
        self.origin_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.cos_yaw = math.cos(self.origin_yaw)
        self.sin_yaw = math.sin(self.origin_yaw)

        self.free = (
            (self.grid >= 0)
            & (self.grid < self.threshold)
        )

        self.map_hash = hashlib.sha256(
            self.grid.tobytes()
        ).hexdigest()

        # Hver cellekant mellom opptatt og fri celle blir en
        # egen veggside. Normalen peker mot den frie cellen.
        faces = [
            # dx, dy, endepunkt A, endepunkt B
            (1, 0, (1, 0), (1, 1)),
            (-1, 0, (0, 0), (0, 1)),
            (0, 1, (0, 1), (1, 1)),
            (0, -1, (0, 0), (1, 0)),
        ]

        occupied_y, occupied_x = np.where(
            self.grid >= self.threshold
        )

        for y, x in zip(occupied_y, occupied_x):
            for dx, dy, a, b in faces:
                nx, ny = x + dx, y + dy

                if not (
                    0 <= nx < self.width
                    and 0 <= ny < self.height
                    and self.free[ny, nx]
                ):
                    continue

                # Koordinater i kartets lokale metersystem.
                p0 = (
                    (x + a[0]) * self.resolution,
                    (y + a[1]) * self.resolution,
                )
                p1 = (
                    (x + b[0]) * self.resolution,
                    (y + b[1]) * self.resolution,
                )
                midpoint = (
                    (p0[0] + p1[0]) / 2.0,
                    (p0[1] + p1[1]) / 2.0,
                )

                self.walls.append({
                    "p0": p0,
                    "p1": p1,
                    "midpoint": midpoint,
                    "normal": (dx, dy),
                })

        self.get_logger().info(
            f"Laget {len(self.walls)} veggsegmenter "
            f"med lengde {self.resolution:.3f} m."
        )
        self.publish_coverage()

    def world_to_local(self, x, y):
        dx = x - self.origin_x
        dy = y - self.origin_y

        return (
            self.cos_yaw * dx + self.sin_yaw * dy,
            -self.sin_yaw * dx + self.cos_yaw * dy,
        )

    def local_to_world(self, x, y):
        return (
            self.origin_x + self.cos_yaw * x - self.sin_yaw * y,
            self.origin_y + self.sin_yaw * x + self.cos_yaw * y,
        )

    def cell_is_free(self, x, y):
        return (
            0 <= x < self.width
            and 0 <= y < self.height
            and bool(self.free[y, x])
        )

    def clear_line(self, start, end):
        """Gå gjennom alle rutenettceller langs siktlinjen.

        Endepunktet ligger rett foran målveggen, i fri plass.
        Ukjente celler behandles som blokkert.
        """
        sx = start[0] / self.resolution
        sy = start[1] / self.resolution
        ex = end[0] / self.resolution
        ey = end[1] / self.resolution

        x, y = math.floor(sx), math.floor(sy)
        end_x, end_y = math.floor(ex), math.floor(ey)

        dx, dy = ex - sx, ey - sy
        step_x = 1 if dx > 0 else -1
        step_y = 1 if dy > 0 else -1

        delta_x = abs(1.0 / dx) if dx != 0 else math.inf
        delta_y = abs(1.0 / dy) if dy != 0 else math.inf

        boundary_x = x + 1 if dx > 0 else x
        boundary_y = y + 1 if dy > 0 else y

        next_x = (
            (boundary_x - sx) / dx if dx != 0 else math.inf
        )
        next_y = (
            (boundary_y - sy) / dy if dy != 0 else math.inf
        )

        while True:
            if not self.cell_is_free(x, y):
                return False

            if x == end_x and y == end_y:
                return True

            if abs(next_x - next_y) < 1e-12:
                # Konservativ behandling av hjørner:
                # begge nabocellene må være frie.
                if (
                    not self.cell_is_free(x + step_x, y)
                    or not self.cell_is_free(x, y + step_y)
                ):
                    return False

                x += step_x
                y += step_y
                next_x += delta_x
                next_y += delta_y

            elif next_x < next_y:
                x += step_x
                next_x += delta_x

            else:
                y += step_y
                next_y += delta_y

    def process_image(self, msg):
        if self.grid is None:
            return

        stamp = Time.from_msg(msg.header.stamp)
        stamp_ns = stamp.nanoseconds

        # TF tolker nulltid som "nyeste"; unngå å bruke det
        # som om det var et tidsstemplet bilde.
        if stamp_ns == 0:
            return

        if self.last_image_stamp is not None:
            elapsed = (
                stamp_ns - self.last_image_stamp
            ) / 1e9
            if 0 <= elapsed < self.update_period:
                return

        self.last_image_stamp = stamp_ns
        frame = self.camera_frame or msg.header.frame_id

        if not frame:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, frame, stamp
            )
        except TransformException as error:
            self.get_logger().warning(
                f"Mangler TF for kamerabildet: {error}",
                throttle_duration_sec=5.0,
            )
            return

        translation = transform.transform.translation
        camera = self.world_to_local(
            translation.x, translation.y
        )

        q = transform.transform.rotation
        norm = math.sqrt(
            q.x * q.x + q.y * q.y
            + q.z * q.z + q.w * q.w
        )
        if norm < 1e-9:
            return

        x, y, z, w = (
            q.x / norm, q.y / norm,
            q.z / norm, q.w / norm,
        )

        # I optical frame peker +Z fremover.
        # Dette er +Z-aksen uttrykt i kartets koordinater.
        forward_x = 2.0 * (x * z + w * y)
        forward_y = 2.0 * (y * z - w * x)

        if math.hypot(forward_x, forward_y) < 1e-6:
            return

        heading = (
            math.atan2(forward_y, forward_x)
            - self.origin_yaw
        )

        visible = set()

        for wall_id, wall in enumerate(self.walls):
            mx, my = wall["midpoint"]
            nx, ny = wall["normal"]

            dx = mx - camera[0]
            dy = my - camera[1]
            distance = math.hypot(dx, dy)

            if not self.min_range <= distance <= self.max_range:
                continue

            angle = (
                math.atan2(dy, dx) - heading + math.pi
            ) % (2.0 * math.pi) - math.pi

            if abs(angle) > self.half_fov:
                continue

            # Vegg → kamera sammenlignes med utadrettet normal.
            # Avviser også baksiden av veggen.
            incidence_cos = (
                -dx * nx - dy * ny
            ) / distance

            if incidence_cos < self.min_incidence_cos:
                continue

            # Flytt målpunktet minimalt ut fra veggen.
            # Da krever vi ikke at selve veggcellen er fri.
            epsilon = self.resolution * 1e-4
            target = (
                mx + nx * epsilon,
                my + ny * epsilon,
            )

            if self.clear_line(camera, target):
                visible.add(wall_id)

        self.visible = visible
        self.seen.update(visible)
        self.last_success_time = self.get_clock().now()
    def on_image(self, msg):
    # Behold bildet vi venter på. Nyere bilder droppes
    # til dette er behandlet eller blitt for gammelt.
        if self.grid is None or self.pending_image is not None:
            return

        if (
            msg.header.stamp.sec == 0
            and msg.header.stamp.nanosec == 0
        ):
            return

        self.pending_image = msg


    def process_pending_image(self):
        msg = self.pending_image
        if msg is None:
            return

        stamp = Time.from_msg(msg.header.stamp)
        frame = self.camera_frame or msg.header.frame_id

        if not frame:
            self.pending_image = None
            return

        age = (
            self.get_clock().now().nanoseconds
            - stamp.nanoseconds
        ) / 1e9

        # Dropp gamle bilder eller bilder fra før en tidsreset.
        if age > 2.0 or age < -0.1:
            self.get_logger().warning(
                "Dropper kamerabilde: TF kom ikke i tide, "
                "eller simuleringstiden ble endret.",
                throttle_duration_sec=5.0,
            )
            self.pending_image = None
            return

        # Ikke blokker ROS mens vi venter. TF-abonnementene
        # får dermed fortsette å motta transformasjoner.
        if not self.tf_buffer.can_transform(
            self.map_frame,
            frame,
            stamp,
            timeout=Duration(seconds=0.0),
        ):
            return

        self.pending_image = None
        self.process_image(msg)

    def publish_coverage(self):
        if self.grid is None:
            return

        current = self.visible

        if self.last_success_time is None:
            current = set()
        else:
            age = (
                self.get_clock().now() - self.last_success_time
            ).nanoseconds / 1e9

            if age < 0 or age > 1.0:
                current = set()

        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "wall_coverage"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.025
        marker.color.a = 1.0

        for wall_id, wall in enumerate(self.walls):
            if wall_id in current:
                color = ColorRGBA(
                    r=0.1, g=0.5, b=1.0, a=1.0
                )
            elif wall_id in self.seen:
                color = ColorRGBA(
                    r=0.1, g=0.9, b=0.2, a=1.0
                )
            else:
                color = ColorRGBA(
                    r=1.0, g=0.15, b=0.1, a=1.0
                )

            for endpoint in (wall["p0"], wall["p1"]):
                wx, wy = self.local_to_world(*endpoint)
                marker.points.append(Point(
                    x=float(wx),
                    y=float(wy),
                    z=self.display_height,
                ))
                marker.colors.append(color)

        self.marker_pub.publish(marker)

    def save_coverage(self, request, response):
        if self.grid is None:
            response.success = False
            response.message = "Har ikke mottatt kart ennå."
            return response

        # JSON inneholder både geometri og historisk dekning.
        segments = []

        for wall_id, wall in enumerate(self.walls):
            segments.append({
                "id": wall_id,
                "start": list(self.local_to_world(*wall["p0"])),
                "end": list(self.local_to_world(*wall["p1"])),
                "seen": wall_id in self.seen,
            })

        data = {
            "version": 1,
            "frame_id": self.map_frame,
            "map_sha256": self.map_hash,
            "map_width": self.width,
            "map_height": self.height,
            "resolution": self.resolution,
            "origin": [
                self.origin_x, self.origin_y, self.origin_yaw
            ],
            "model": {
                "horizontal_fov_deg": math.degrees(
                    2.0 * self.half_fov
                ),
                "min_range": self.min_range,
                "max_range": self.max_range,
                "max_incidence_deg": math.degrees(
                    math.acos(self.min_incidence_cos)
                ),
            },
            "seen_segments": len(self.seen),
            "total_segments": len(self.walls),
            "segments": segments,
        }

        try:
            output = FilePath(self.save_path).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)

            # Skriv midlertidig fil først, erstatt deretter.
            temporary = output.with_name(output.name + ".tmp")
            temporary.write_text(
                json.dumps(data, indent=2),
                encoding="utf-8",
            )
            temporary.replace(output)

            response.success = True
            response.message = (
                f"Lagret {len(self.seen)}/{len(self.walls)} "
                f"segmenter til {output}"
            )
        except OSError as error:
            response.success = False
            response.message = str(error)

        return response


def main(args=None):
    rclpy.init(args=args)
    node = WallCoverage()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()