import math
import random
from collections import defaultdict

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import qmc


class PRM:
    def __init__(
        self,
        grid,
        resolution=1.0,
        origin=(0.0, 0.0),
        origin_yaw=0.0,
        cell_size=2.0,
        points_per_cell=12,
        neighbors=12,
        max_edge_length=4.0,
        robot_radius=0.22,
        obstacle_threshold=50,
        seed=42,
    ):
        self.grid = np.asarray(grid, dtype=np.int16)

        if self.grid.ndim != 2 or self.grid.size == 0:
            raise ValueError(
                'grid kan ikke være tomt 2D-rutenett'
            )

        if resolution <= 0 or cell_size <= 0 or max_edge_length <= 0:
            raise ValueError(
                'resolution, cell_size og max_edge_length må være positive'
            )

        if robot_radius < 0 or points_per_cell < 1 or neighbors < 1:
            raise ValueError(
                'robot_radius må være >= 0, og punkter/naboer må være >= 1'
            )

        self.width = self.grid.shape[1]
        self.height = self.grid.shape[0]

        self.resolution = float(resolution)
        self.origin = tuple(origin)
        self.origin_yaw = float(origin_yaw)

        self._cos_yaw = math.cos(self.origin_yaw)
        self._sin_yaw = math.sin(self.origin_yaw)

        self.cell_size = float(cell_size)
        self.points_per_cell = int(points_per_cell)
        self.neighbors = int(neighbors)
        self.max_edge_length = float(max_edge_length)
        self.robot_radius = float(robot_radius)
        self.obstacle_threshold = obstacle_threshold
        self.seed = seed

        self.nodes = []
        self.edges = {}
        self.cell_nodes = {}

        self.free = self._make_free_grid()

        self.generate()

    @classmethod
    def from_occupancy_grid(cls, msg, **kwargs):
        """Lag PRM fra en ROS OccupancyGrid."""

        width = msg.info.width
        height = msg.info.height

        if len(msg.data) != width * height:
            raise ValueError('OccupancyGrid har ugyldig størrelse')

        grid = np.asarray(
            msg.data,
            dtype=np.int16
        ).reshape((height, width))

        q = msg.info.origin.orientation

        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

        return cls(
            grid,
            resolution=msg.info.resolution,
            origin=(
                msg.info.origin.position.x,
                msg.info.origin.position.y
            ),
            origin_yaw=yaw,
            **kwargs,
        )

    def grid_to_world(self, x, y):
        gx = (x + 0.5) * self.resolution
        gy = (y + 0.5) * self.resolution

        return (
            self.origin[0]
            + self._cos_yaw * gx
            - self._sin_yaw * gy,

            self.origin[1]
            + self._sin_yaw * gx
            + self._cos_yaw * gy
        )

    def world_to_grid(self, x, y):
        dx = x - self.origin[0]
        dy = y - self.origin[1]

        return (
            math.floor(
                (self._cos_yaw * dx + self._sin_yaw * dy)
                / self.resolution
            ),
            math.floor(
                (-self._sin_yaw * dx + self._cos_yaw * dy)
                / self.resolution
            )
        )

    def _make_free_grid(self):
        safe = (
            (self.grid >= 0)
            & (self.grid < self.obstacle_threshold)
        )

        if self.robot_radius == 0:
            return safe

        margin = (
            self.robot_radius
            + self.resolution * math.sqrt(2) / 2
        )

        # Behandle utsiden av kartet som en hindring.
        padded = np.pad(
            safe,
            1,
            constant_values=False
        )

        # Beregn avstanden til nærmeste hindring.
        distances = distance_transform_edt(padded)[1:-1, 1:-1]
        distances *= self.resolution

        # Fjern celler som ligger for nær hindringer.
        safe &= distances > margin

        return safe

    def is_free(self, x, y):
        gx, gy = self.world_to_grid(x, y)

        return (
            0 <= gx < self.width
            and 0 <= gy < self.height
            and self.free[gy, gx]
        )

    def line_is_free(self, start, end):
        distance = math.dist(start, end)

        steps = max(
            1,
            math.ceil(distance / (self.resolution / 2.0))
        )

        t = np.linspace(0.0, 1.0, steps + 1)

        x = start[0] + (end[0] - start[0]) * t
        y = start[1] + (end[1] - start[1]) * t

        dx = x - self.origin[0]
        dy = y - self.origin[1]

        gx = np.floor(
            (
                self._cos_yaw * dx
                + self._sin_yaw * dy
            ) / self.resolution
        ).astype(int)

        gy = np.floor(
            (
                -self._sin_yaw * dx
                + self._cos_yaw * dy
            ) / self.resolution
        ).astype(int)

        valid = (
            (gx >= 0)
            & (gx < self.width)
            & (gy >= 0)
            & (gy < self.height)
        )

        if not np.all(valid):
            return False

        return bool(np.all(self.free[gy, gx]))

    def generate(self):
        self.nodes = []
        self.edges = {}
        self.cell_nodes = {}

        size = max(1, math.ceil(self.cell_size / self.resolution))

        engine = qmc.PoissonDisk(d=2, radius=0.25, seed=self.seed)

        unit_points = engine.fill_space()

        for pt in unit_points:
            x = int(pt[0] * self.width)
            y = int(pt[1] * self.height)

            x = min(max(x, 0), self.width - 1)
            y = min(max(y, 0), self.height - 1)

            if not self.free[y, x]:
                continue

            cell = (x // size, y // size)

            if cell not in self.cell_nodes:
                self.cell_nodes[cell] = []

            if len(self.cell_nodes[cell]) >= self.points_per_cell:
                continue

            node_id = len(self.nodes)
            self.nodes.append(self.grid_to_world(x, y))
            self.edges[node_id] = {}
            self.cell_nodes[cell].append(node_id)

        self._connect_nodes()

        return self

    def _connect_nodes(self):
        bins = defaultdict(list)
        spacing = self.max_edge_length

        for node_id, (x, y) in enumerate(self.nodes):
            bins[
                (
                    math.floor(x / spacing),
                    math.floor(y / spacing)
                )
            ].append(node_id)

        for node_id, point in enumerate(self.nodes):
            bx = math.floor(point[0] / spacing)
            by = math.floor(point[1] / spacing)

            candidates = []

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nearby = bins.get(
                        (bx + dx, by + dy),
                        []
                    )

                    for other_id in nearby:
                        if other_id == node_id:
                            continue

                        distance = math.dist(
                            point,
                            self.nodes[other_id]
                        )

                        if distance <= self.max_edge_length:
                            candidates.append(
                                (distance, other_id)
                            )

            for distance, other_id in sorted(candidates)[
                :self.neighbors
            ]:
                if other_id in self.edges[node_id]:
                    continue

                if self.line_is_free(
                    point,
                    self.nodes[other_id]
                ):
                    self.edges[node_id][other_id] = distance
                    self.edges[other_id][node_id] = distance

    def nearest_node(self, x, y, visible_only=True):
        """Finn nærmeste PRM-node."""

        point = (x, y)

        if visible_only and not self.is_free(x, y):
            return None

        candidates = sorted(
            range(len(self.nodes)),
            key=lambda i: math.dist(point, self.nodes[i])
        )

        for node_id in candidates:
            if (
                not visible_only
                or self.line_is_free(point, self.nodes[node_id])
            ):
                return node_id

        return None