import math
import heapq

class AStar:
    @staticmethod
    def find_path(prm, start_node_id, goal_node_id, preferred_wall_dist=0.6):
        if start_node_id not in prm.edges or goal_node_id not in prm.edges:
            return None

        open_set = []
        heapq.heappush(open_set, (0.0, start_node_id))
        came_from = {}
        
        g_score = {node_id: float('inf') for node_id in range(len(prm.nodes))}
        g_score[start_node_id] = 0.0
        
        f_score = {node_id: float('inf') for node_id in range(len(prm.nodes))}
        f_score[start_node_id] = AStar._heuristic(prm, start_node_id, goal_node_id)
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal_node_id:
                return AStar._reconstruct_path(came_from, current, prm)
            
            for neighbor, distance in prm.edges[current].items():
                nx, ny = prm.nodes[neighbor]
                gx, gy = prm.world_to_grid(nx, ny)
                
                clearance = prm.resolution * 5
                if hasattr(prm, 'clearance_grid') and 0 <= gx < prm.width and 0 <= gy < prm.height:
                    clearance = prm.clearance_grid[gy, gx]
                
                wall_penalty = abs(clearance - preferred_wall_dist) * 2.0
                tentative_g_score = g_score[current] + distance + wall_penalty
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + AStar._heuristic(prm, neighbor, goal_node_id)
                    
                    if not any(neighbor == item[1] for item in open_set):
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        
        return None

    @staticmethod
    def _heuristic(prm, node_a, node_b):
        x1, y1 = prm.nodes[node_a]
        x2, y2 = prm.nodes[node_b]
        return math.dist((x1, y1), (x2, y2))

    @staticmethod
    def _reconstruct_path(came_from, current, prm):
        path = [prm.nodes[current]]
        while current in came_from:
            current = came_from[current]
            path.append(prm.nodes[current])
        path.reverse()
        return path