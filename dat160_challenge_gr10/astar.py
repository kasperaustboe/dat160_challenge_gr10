import heapq
import math


class AStar:
    @staticmethod
    def find_path(prm, start_node_id, goal_node_id, preferred_wall_dist=0.6):
        """Finn en rute i PRM-grafen. Returnerer en liste med (x, y) eller None."""
        if start_node_id not in prm.edges or goal_node_id not in prm.edges:
            return None

        # PRM har ingen avstandskart per node, så avstanden til veggen anslås
        # som fem celler. Straffen blir dermed lik for alle kanter og gjør at
        # ruter med færre veipunkter foretrekkes.
        estimated_clearance = prm.resolution * 5
        edge_penalty = abs(estimated_clearance - preferred_wall_dist) * 2.0

        goal = prm.nodes[goal_node_id]
        g_score = {start_node_id: 0.0}
        came_from = {}
        open_set = [(math.dist(prm.nodes[start_node_id], goal), start_node_id)]

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_node_id:
                return AStar._reconstruct_path(prm, came_from, current)

            for neighbor, distance in prm.edges[current].items():
                cost = g_score[current] + distance + edge_penalty
                if cost < g_score.get(neighbor, math.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = cost
                    heapq.heappush(open_set, (cost + math.dist(prm.nodes[neighbor], goal), neighbor))

        return None

    @staticmethod
    def _reconstruct_path(prm, came_from, current):
        path = [prm.nodes[current]]
        while current in came_from:
            current = came_from[current]
            path.append(prm.nodes[current])
        path.reverse()
        return path
