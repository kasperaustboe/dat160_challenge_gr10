import heapq
import math
from collections import defaultdict

import numpy as np
from scipy.ndimage import distance_transform_edt, label
from scipy.stats import qmc


def quaternion_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


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
        reachable_from=None,
    ):
        self.grid = np.asarray(grid, dtype=np.int16)
        if self.grid.ndim != 2 or self.grid.size == 0:
            raise ValueError('grid kan ikke være tomt 2D-rutenett')
        if resolution <= 0 or cell_size <= 0 or max_edge_length <= 0:
            raise ValueError('resolution, cell_size og max_edge_length må være positive')
        if robot_radius < 0 or points_per_cell < 1 or neighbors < 1:
            raise ValueError('robot_radius må være >= 0, og punkter/naboer må være >= 1')

        self.height, self.width = self.grid.shape
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

        # navigable: alle celler med nok avstand til vegg.
        # free: de av dem som henger sammen med en startposisjon.
        self._wall_distances = None
        self.navigable = self._make_free_grid()
        self.free = self.navigable
        if reachable_from is not None:
            self.free = self._reachable_free_grid(reachable_from)

        self.generate()

    @classmethod
    def from_occupancy_grid(cls, msg, **kwargs):
        """Lag PRM fra en ROS OccupancyGrid."""
        width, height = msg.info.width, msg.info.height
        if len(msg.data) != width * height:
            raise ValueError('OccupancyGrid har ugyldig størrelse')

        grid = np.asarray(msg.data, dtype=np.int16).reshape((height, width))
        origin = msg.info.origin
        return cls(
            grid,
            resolution=msg.info.resolution,
            origin=(origin.position.x, origin.position.y),
            origin_yaw=quaternion_to_yaw(origin.orientation),
            **kwargs,
        )

    # --- Koordinater ---

    def grid_to_world(self, x, y):
        gx = (x + 0.5) * self.resolution
        gy = (y + 0.5) * self.resolution
        return (
            self.origin[0] + self._cos_yaw * gx - self._sin_yaw * gy,
            self.origin[1] + self._sin_yaw * gx + self._cos_yaw * gy,
        )

    def world_to_grid(self, x, y):
        gx, gy = self._world_to_grid_float(x, y)
        return math.floor(gx), math.floor(gy)

    def _world_to_grid_float(self, x, y):
        """Virker både for tall og numpy-arrays."""
        dx = x - self.origin[0]
        dy = y - self.origin[1]
        return (
            (self._cos_yaw * dx + self._sin_yaw * dy) / self.resolution,
            (-self._sin_yaw * dx + self._cos_yaw * dy) / self.resolution,
        )

    def in_bounds(self, gx, gy):
        return 0 <= gx < self.width and 0 <= gy < self.height

    # --- Ledig område ---

    def _make_free_grid(self):
        safe = (self.grid >= 0) & (self.grid < self.obstacle_threshold)
        if self.robot_radius == 0:
            return safe

        margin = self.robot_radius + self.resolution * math.sqrt(2) / 2
        # Utsiden av kartet behandles som hindring.
        padded = np.pad(safe, 1, constant_values=False)
        distances = distance_transform_edt(padded)[1:-1, 1:-1] * self.resolution
        self._wall_distances = distances
        # Fjern celler som ligger for nær hindringer.
        return safe & (distances > margin)

    def _reachable_free_grid(self, spawn_points):
        # Nummerer områdene som henger sammen via frie naboceller.
        components, _ = label(self.free)
        spawn_components = set()
        for x, y in spawn_points:
            gx, gy = self.world_to_grid(x, y)
            if self.in_bounds(gx, gy) and self.free[gy, gx]:
                spawn_components.add(components[gy, gx])

        if not spawn_components:
            raise ValueError('Ingen startposisjoner ligger i et ledig PRM-område')
        return np.isin(components, list(spawn_components))

    def is_free(self, x, y):
        gx, gy = self.world_to_grid(x, y)
        return self.in_bounds(gx, gy) and bool(self.free[gy, gx])

    def line_is_free(self, start, end, *, leave_margin=False, unfiltered=False):
        distance = math.dist(start, end)
        steps = max(1, math.ceil(distance / (self.resolution / 2.0)))
        t = np.linspace(0.0, 1.0, steps + 1)
        gx, gy = self._world_to_grid_float(
            start[0] + (end[0] - start[0]) * t,
            start[1] + (end[1] - start[1]) * t,
        )
        gx = np.floor(gx).astype(int)
        gy = np.floor(gy).astype(int)

        if not np.all((gx >= 0) & (gx < self.width) & (gy >= 0) & (gy < self.height)):
            return False

        mask = self.navigable if unfiltered else self.free
        free = mask[gy, gx]
        if np.all(free):
            return True
        if (not leave_margin or self._wall_distances is None or distance > 0.45
                or free[0] or not free[-1]):
            return False

        # Roboten kan allerede stå innenfor marginen. La den kjøre en kort vei
        # bort fra veggen, med vanlig sikkerhetsmargin så snart den er ute.
        raw = self.grid[gy, gx]
        first_free = np.argmax(free)
        clearance = self._wall_distances[gy[:first_free + 1], gx[:first_free + 1]]
        return bool(np.all((raw >= 0) & (raw < self.obstacle_threshold))
                    and np.all(free[first_free:]) and np.all(np.diff(clearance) >= 0))

    # --- Rutesøk direkte i kartcellene ---

    def shortest_grid_route(self, start, goals):
        """Finn en rute i kartcellene uten å bruke PRM-grafen."""
        free = self.navigable
        if not goals:
            return None
        sx, sy = self.world_to_grid(*start)
        if not self.in_bounds(sx, sy):
            return None

        if free[sy, sx]:
            costs = {(sx, sy): 0.0}
        else:
            # Bruk samme kontrollerte utgang fra sikkerhetsmarginen som i PRM-søket.
            radius = math.ceil(0.45 / self.resolution)
            costs = {}
            for y in range(max(0, sy - radius), min(self.height, sy + radius + 1)):
                for x in range(max(0, sx - radius), min(self.width, sx + radius + 1)):
                    point = self.grid_to_world(x, y)
                    if free[y, x] and self.line_is_free(start, point, leave_margin=True, unfiltered=True):
                        costs[(x, y)] = math.dist(start, point)

        goal_x, goal_y = zip(*goals)
        xmin, xmax, ymin, ymax = min(goal_x), max(goal_x), min(goal_y), max(goal_y)

        def estimate(x, y):
            # Avstand til boksen rundt alle målene.
            return self.resolution * math.hypot(max(xmin - x, 0, x - xmax), max(ymin - y, 0, y - ymax))

        steps = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
        queue = [(cost + estimate(*cell), cost, cell) for cell, cost in costs.items()]
        heapq.heapify(queue)
        previous = {}
        while queue:
            _, cost, cell = heapq.heappop(queue)
            if cost != costs[cell]:
                continue
            if cell in goals:
                return [start] + self._route_corners(previous, cell)
            x, y = cell
            for dx, dy in steps:
                nx, ny = x + dx, y + dy
                if not (self.in_bounds(nx, ny) and free[ny, nx]):
                    continue
                # Ikke kutt hjørner diagonalt.
                if dx and dy and (not free[y, nx] or not free[ny, x]):
                    continue
                new_cost = cost + self.resolution * math.hypot(dx, dy)
                if new_cost < costs.get((nx, ny), math.inf):
                    costs[(nx, ny)] = new_cost
                    previous[(nx, ny)] = cell
                    heapq.heappush(queue, (new_cost + estimate(nx, ny), new_cost, (nx, ny)))
        return None

    def _route_corners(self, previous, cell):
        cells = [cell]
        while cell in previous:
            cell = previous[cell]
            cells.append(cell)
        cells.reverse()
        # Behold bare punktene der retningen endres, pluss første og siste celle.
        corners = [b for a, b, c in zip(cells, cells[1:], cells[2:])
                   if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1])]
        return [self.grid_to_world(*p) for p in (cells[0], *corners, cells[-1])]

    # --- Bygging av PRM-grafen ---

    def generate(self):
        self.nodes = []
        self.edges = {}
        cell_nodes = defaultdict(int)
        size = max(1, math.ceil(self.cell_size / self.resolution))

        engine = qmc.PoissonDisk(d=2, radius=0.025, seed=self.seed)
        for px, py in engine.fill_space():
            x = min(max(int(px * self.width), 0), self.width - 1)
            y = min(max(int(py * self.height), 0), self.height - 1)
            if not self.free[y, x]:
                continue

            # Begrens antall noder per celle for jevn fordeling.
            cell = (x // size, y // size)
            if cell_nodes[cell] >= self.points_per_cell:
                continue
            cell_nodes[cell] += 1

            self.edges[len(self.nodes)] = {}
            self.nodes.append(self.grid_to_world(x, y))

        self._connect_nodes()
        return self

    def _connect_nodes(self):
        spacing = self.max_edge_length

        def bin_of(point):
            return math.floor(point[0] / spacing), math.floor(point[1] / spacing)

        bins = defaultdict(list)
        for node_id, point in enumerate(self.nodes):
            bins[bin_of(point)].append(node_id)

        for node_id, point in enumerate(self.nodes):
            bx, by = bin_of(point)
            candidates = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for other_id in bins.get((bx + dx, by + dy), ()):
                        if other_id == node_id:
                            continue
                        distance = math.dist(point, self.nodes[other_id])
                        if distance <= self.max_edge_length:
                            candidates.append((distance, other_id))

            for distance, other_id in sorted(candidates)[:self.neighbors]:
                if other_id in self.edges[node_id]:
                    continue
                if self.line_is_free(point, self.nodes[other_id]):
                    self.edges[node_id][other_id] = distance
                    self.edges[other_id][node_id] = distance
