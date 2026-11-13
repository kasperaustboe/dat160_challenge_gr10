"""Reserverer ruter slik at robotene ikke kjører inn i hverandre."""
import copy
import heapq
import math


def in_y_zone(point, side):
    return side is None or side * point[1] > 0.0


def point_segment_distance(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length2 = dx * dx + dy * dy
    if length2 == 0:
        return math.dist(point, start)
    t = max(0.0, min(1.0, ((point[0] - start[0]) * dx +
                           (point[1] - start[1]) * dy) / length2))
    return math.dist(point, (start[0] + t * dx, start[1] + t * dy))


def segment_distance(a, b, c, d):
    def cross(p, q, r):
        return ((q[0] - p[0]) * (r[1] - p[1]) -
                (q[1] - p[1]) * (r[0] - p[0]))

    ab_c, ab_d = cross(a, b, c), cross(a, b, d)
    cd_a, cd_b = cross(c, d, a), cross(c, d, b)
    if ab_c * ab_d < 0 and cd_a * cd_b < 0:
        return 0.0
    return min(point_segment_distance(a, c, d), point_segment_distance(b, c, d),
               point_segment_distance(c, a, b), point_segment_distance(d, a, b))


class RouteReservations:
    def __init__(self, clearance=0.75):
        self.clearance = clearance
        self.routes = {}
        self._cache = {}

    def reserve(self, robot_id, points):
        self.routes[robot_id] = tuple(points)
        self._cache.clear()

    def release(self, robot_id):
        self.routes.pop(robot_id, None)
        self._cache.clear()

    def advance(self, robot_id, position, tolerance=0.10):
        """Frigi passerte veipunkter; behold bare ruten foran roboten."""
        points = self.routes.get(robot_id)
        if not points:
            return
        # Første punkt er forrige kjente robotposisjon, ikke et fremtidig mål.
        remaining = list(points[1:])
        while remaining and math.dist(position, remaining[0]) <= tolerance:
            remaining.pop(0)
        if remaining:
            self.routes[robot_id] = (tuple(position),) + tuple(remaining)
        else:
            self.routes.pop(robot_id, None)
        self._cache.clear()

    def segments(self, robot_id, positions, extra_routes=None, *, ignore_robots=()):
        routes = dict(self.routes)
        routes.update(extra_routes or {})
        result = []
        for other in set(positions) | set(routes):
            if other == robot_id or other in ignore_robots:
                continue
            position = positions.get(other)
            if position is not None:
                # En stillestående robot opptar også fysisk plass.
                result.append((position, position))
            points = routes.get(other, ())
            if len(points) == 1:
                result.append((points[0], points[0]))
            result.extend(zip(points, points[1:]))
        return result

    def clear_segment(self, start, end, segments):
        margin = self.clearance
        for a, b in segments:
            # Billig avvisning før avstandsberegning.
            if (max(start[0], end[0]) + margin < min(a[0], b[0]) or
                    max(a[0], b[0]) + margin < min(start[0], end[0]) or
                    max(start[1], end[1]) + margin < min(a[1], b[1]) or
                    max(a[1], b[1]) + margin < min(start[1], end[1])):
                continue
            if segment_distance(start, end, a, b) < margin:
                return False
        return True

    def shortest_routes(self, prm, robot_id, start, goals, positions, extra_routes=None, *,
                        ignore_robots=(), allow_start_escape=False):
        """Gi én korteste rute per nåbart mål, sortert etter kjørelengde."""
        occupied = self.segments(robot_id, positions, extra_routes, ignore_robots=ignore_robots)
        start_free = prm.is_free(*start)
        escape = allow_start_escape and not start_free
        if (not start_free and not escape) or not self.clear_segment(start, start, occupied):
            return
        goals = [goal for goal in dict.fromkeys(goals) if prm.is_free(*goal)
                 and self.clear_segment(goal, goal, occupied)]
        if not goals:
            return

        start_id = len(prm.nodes)
        points = [*prm.nodes, start, *goals]
        goal_ids = range(start_id + 1, len(points))
        remaining_goals = len(goals)
        distances = {start_id: 0.0}
        previous = {}
        queue = [(0.0, start_id)]

        while queue:
            distance, current = heapq.heappop(queue)
            if distance != distances[current]:
                continue
            if current in goal_ids:
                route = [points[current]]
                while current != start_id:
                    current = previous[current]
                    route.append(points[current])
                yield route[::-1]
                remaining_goals -= 1
                if not remaining_goals:
                    return
                continue

            point = points[current]
            neighbors = range(start_id) if current == start_id else prm.edges.get(current, ())
            for neighbor in (*goal_ids, *neighbors):
                end = points[neighbor]
                length = math.dist(point, end)
                candidate = distance + length
                if candidate >= distances.get(neighbor, math.inf):
                    continue
                # PRM-kantene er allerede sjekket mot kartet. Nye koblinger må sjekkes her.
                if current == start_id or neighbor in goal_ids:
                    direct = current == start_id and neighbor in goal_ids
                    if not direct and length > prm.max_edge_length:
                        continue
                    if escape and current == start_id:
                        clear = prm.line_is_free(point, end, leave_margin=True)
                    else:
                        clear = prm.line_is_free(point, end)
                    if not clear:
                        continue
                if self.clear_segment(point, end, occupied):
                    distances[neighbor] = candidate
                    previous[neighbor] = current
                    heapq.heappush(queue, (candidate, neighbor))

    def plan(self, prm, astar, robot_id, start, goal, positions, extra_routes=None, *,
             zone_side=None, ignore_robots=()):
        # Besøkte mål inngår ikke i sperringen. Bare aktive ruter og robotenes
        # nåværende posisjoner brukes her.
        if not in_y_zone(goal, zone_side):
            return None
        returning_to_zone = not in_y_zone(start, zone_side)
        if not prm.is_free(*start) or not prm.is_free(*goal):
            return None
        routes = dict(self.routes)
        routes.update(extra_routes or {})
        key = (id(prm), robot_id, zone_side, returning_to_zone, frozenset(ignore_robots),
               tuple(sorted((r, tuple(p) if p is not None else None) for r, p in positions.items())),
               tuple(sorted((r, tuple(points)) for r, points in routes.items())))
        cached = self._cache.get(key)
        occupied = cached[2] if cached is not None else self.segments(
            robot_id, positions, extra_routes, ignore_robots=ignore_robots,
        )
        if (not self.clear_segment(start, start, occupied) or
                not self.clear_segment(goal, goal, occupied)):
            return None
        if cached is None:
            allowed = {
                i for i, point in enumerate(prm.nodes)
                if (returning_to_zone or in_y_zone(point, zone_side))
                and self.clear_segment(point, point, occupied)
            }
            graph = copy.copy(prm)
            graph.edges = {i: {} for i in allowed}
            edge_is_clear = {}
            for i in allowed:
                for j, cost in prm.edges.get(i, {}).items():
                    if j not in allowed:
                        continue
                    # Etter redning kan roboten krysse tilbake til sin sone én gang.
                    if in_y_zone(prm.nodes[i], zone_side) and not in_y_zone(prm.nodes[j], zone_side):
                        continue
                    edge = (min(i, j), max(i, j))
                    if edge not in edge_is_clear:
                        edge_is_clear[edge] = self.clear_segment(prm.nodes[i], prm.nodes[j], occupied)
                    if edge_is_clear[edge]:
                        graph.edges[i][j] = cost
            if len(self._cache) >= 16:
                self._cache.clear()
            self._cache[key] = (graph, allowed, occupied)
        else:
            graph, allowed, _ = cached

        def nearest(point):
            point_in_zone = in_y_zone(point, zone_side)
            for i in sorted(allowed, key=lambda n: math.dist(point, prm.nodes[n])):
                if (in_y_zone(prm.nodes[i], zone_side) == point_in_zone and
                        prm.line_is_free(point, prm.nodes[i]) and
                        self.clear_segment(point, prm.nodes[i], occupied)):
                    return i
            return None

        first, last = nearest(start), nearest(goal)
        if first is None or last is None:
            return None
        points = astar.find_path(graph, first, last)
        if not points:
            return None
        # Reserver også forbindelsen fra faktisk robotposisjon og helt til mål.
        result = [tuple(start)] + list(points)
        if math.dist(result[-1], goal) > 0.001:
            result.append(tuple(goal))
        return result
