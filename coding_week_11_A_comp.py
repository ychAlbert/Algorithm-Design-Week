import numpy as np
import random
import tkinter as tk
from datetime import datetime
from tkinter import simpledialog, filedialog
from PIL import ImageGrab
import heapq


class Maze:
    def __init__(self, size=8, obstacle_prob=0.3):
        self.size = size
        self.obstacle_prob = obstacle_prob
        self.maze = self.generate_maze()
        self.path = []

    def generate_maze(self):
        maze = np.zeros((self.size, self.size), dtype=int)
        for i in range(self.size):
            for j in range(self.size):
                if random.random() < self.obstacle_prob:
                    maze[i][j] = 1
        maze[0][0] = 0  # 起点
        maze[self.size - 1][self.size - 1] = 0  # 终点
        return maze.tolist()

    def is_valid_move(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size and self.maze[x][y] == 0

    def solve_maze(self, draw_callback, algorithm='A*'):
        self.path = []
        if algorithm == 'A*':
            solver = AStarSolver(self, draw_callback)
        elif algorithm == 'BFS':
            solver = BFSSolver(self, draw_callback)
        elif algorithm == 'DFS':
            solver = DFSSolver(self, draw_callback)
        elif algorithm == 'Greedy':
            solver = GreedySolver(self, draw_callback)
        elif algorithm == 'Bidirectional':
            solver = BidirectionalSolver(self, draw_callback)
        elif algorithm == 'IDDFS':
            solver = IDDFSSolver(self, draw_callback)
        elif algorithm == 'Dijkstra':
            solver = DijkstraSolver(self, draw_callback)
        elif algorithm == 'JPSS':
            solver = JPSSolver(self, draw_callback)
        elif algorithm == 'RRT':
            solver = RRTSolver(self, draw_callback)
        elif algorithm == 'DWA*':
            solver = DWASolver(self, draw_callback)
        elif algorithm == 'RRT*':
            solver = RRTStarSolver(self, draw_callback)
        elif algorithm == 'Hybrid A*':
            solver = HybridAStarSolver(self, draw_callback)
        elif algorithm == 'Potential Field':
            solver = PotentialFieldSolver(self, draw_callback)
        elif algorithm == 'Theta*':
            solver = ThetaStarSolver(self, draw_callback)
        else:
            raise ValueError(f"未知的算法: {algorithm}")
        result = solver.solve()
        return result


class ThetaStarSolver:
    def __init__(self, maze, draw_callback):
        self.maze = maze
        self.draw_callback = draw_callback
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (-1, -1), (1, 1), (1, -1)]

    def heuristic(self, a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def line_of_sight(self, s_start, s_end):
        x0, y0 = s_start
        x1, y1 = s_end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while (x0, y0) != (x1, y1):
            if not self.maze.is_valid_move(x0, y0):
                return False
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return self.maze.is_valid_move(x1, y1)

    def solve(self):
        start = (0, 0)
        end = (self.maze.size - 1, self.maze.size - 1)
        return self.theta_star(start, end)

    def theta_star(self, start, end):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {start: None}
        g_score = {start: 0}

        parent = {start: start}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                self.reconstruct_path(parent, current)
                return self.maze.path

            for direction in self.directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                if not self.maze.is_valid_move(neighbor[0], neighbor[1]):
                    continue

                if self.line_of_sight(parent[current], neighbor):
                    tentative_g_score = g_score[parent[current]] + self.heuristic(parent[current], neighbor)
                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self.heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score, neighbor))
                        parent[neighbor] = parent[current]
                        self.maze.maze[neighbor[0]][neighbor[1]] = 2
                        self.draw_callback(self.maze, neighbor[0], neighbor[1], "green")
                else:
                    tentative_g_score = g_score[current] + self.heuristic(current, neighbor)
                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self.heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score, neighbor))
                        parent[neighbor] = current
                        self.maze.maze[neighbor[0]][neighbor[1]] = 2
                        self.draw_callback(self.maze, neighbor[0], neighbor[1], "green")

        return None

    def reconstruct_path(self, came_from, current):
        while current is not None:
            self.maze.path.append(current)
            self.draw_callback(self.maze, current[0], current[1], "blue")
            current = came_from[current]
        self.maze.path.reverse()
class PotentialFieldSolver:
    def __init__(self, maze, draw_callback):
        self.maze = maze
        self.draw_callback = draw_callback
        self.repulsive_strength = 5.0
        self.attractive_strength = 1.0

    def solve(self):
        start = (0, 0)
        end = (self.maze.size - 1, self.maze.size - 1)
        return self.potential_field(start, end)

    def potential_field(self, start, end):
        current = start
        path = [current]
        self.maze.maze[current[0]][current[1]] = 2  # Mark as visited
        self.draw_callback(self.maze, current[0], current[1], "green")

        while current != end:
            next_move = self.calculate_move(current, end)
            if not next_move:
                return None  # No valid move; path construction failed
            current = next_move
            path.append(current)
            self.maze.maze[current[0]][current[1]] = 2  # Mark as visited
            self.draw_callback(self.maze, current[0], current[1], "green")

        self.maze.path = path
        return path

    def calculate_move(self, current, end):
        min_potential = float('inf')
        best_move = None

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (-1, -1), (1, 1), (1, -1)]
        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if not self.maze.is_valid_move(neighbor[0], neighbor[1]):
                continue

            potential = self.attractive(neighbor, end) + self.repulsive(neighbor)
            if potential < min_potential:
                min_potential = potential
                best_move = neighbor

        return best_move

    def attractive(self, node, end):
        return self.attractive_strength * self.distance(node, end)

    def repulsive(self, node):
        repulsive_force = 0
        for i in range(self.maze.size):
            for j in range(self.maze.size):
                if self.maze.maze[i][j] == 1:
                    distance = self.distance((i, j), node)
                    if distance == 0:
                        return float('inf')
                    repulsive_force += self.repulsive_strength / distance ** 2

        return repulsive_force

    def distance(self, a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class HybridAStarSolver:
    def __init__(self, maze, draw_callback):
        self.maze = maze
        self.draw_callback = draw_callback
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (-1, -1), (1, 1), (1, -1)]
        self.turning_weight = 1.2  # Penalty for turning

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def solve(self):
        start = (0, 0)
        end = (self.maze.size - 1, self.maze.size - 1)
        return self.hybrid_a_star(start, end)

    def hybrid_a_star(self, start, end):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {start: None}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                self.reconstruct_path(came_from, current)
                return self.maze.path

            for direction in self.directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                if not self.maze.is_valid_move(neighbor[0], neighbor[1]):
                    continue

                # Calculate the turning factor
                turning_factor = self.turning_weight if came_from[current] and direction != came_from[current] else 1

                tentative_g_score = g_score[current] + turning_factor

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = direction
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, neighbor))
                    self.maze.maze[neighbor[0]][neighbor[1]] = 2  # Mark as visited
                    self.draw_callback(self.maze, neighbor[0], neighbor[1], "green")

        return None

    def reconstruct_path(self, came_from, current):
        while current is not None:
            self.maze.path.append(current)
            self.draw_callback(self.maze, current[0], current[1], "blue")
            current = came_from[current]
        self.maze.path.reverse()


class RRTStarSolver:
    def __init__(self, maze, draw_callback):
        self.maze = maze
        self.draw_callback = draw_callback
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (-1, -1), (1, 1), (1, -1)]
        self.step_size = 1.0
        self.radius = 2.0

    def solve(self):
        start = (0, 0)
        end = (self.maze.size - 1, self.maze.size - 1)
        return self.rrt_star(start, end)

    def rrt_star(self, start, end):
        tree = {start: None}
        nodes = [start]
        costs = {start: 0}

        while True:
            random_point = self.random_sample()
            nearest_node = self.nearest(tree, random_point)
            new_node = self.steer(nearest_node, random_point)

            if self.maze.is_valid_move(new_node[0], new_node[1]):
                tree[new_node] = nearest_node
                nodes.append(new_node)

                costs[new_node] = costs[nearest_node] + self.distance(nearest_node, new_node)

                neighborhood = self.get_neighborhood(nodes, new_node)
                for neighbor in neighborhood:
                    if costs[neighbor] + self.distance(new_node, neighbor) < costs[new_node]:
                        tree[new_node] = neighbor
                        costs[new_node] = costs[neighbor] + self.distance(new_node, neighbor)

                self.maze.maze[new_node[0]][new_node[1]] = 2  # Mark as visited
                self.draw_callback(self.maze, new_node[0], new_node[1], "green")

                if self.heuristic(new_node, end) <= self.step_size:
                    self.reconstruct_path(tree, new_node)
                    self.reconstruct_path(tree, end, partial=True)
                    return self.maze.path

        return None

    def random_sample(self):
        return (random.randint(0, self.maze.size - 1), random.randint(0, self.maze.size - 1))

    def nearest(self, tree, sample):
        return min(tree.keys(), key=lambda node: self.heuristic(node, sample))

    def steer(self, from_node, to_node):
        from_x, from_y = from_node
        to_x, to_y = to_node
        dx, dy = to_x - from_x, to_y - from_y
        distance = self.heuristic(from_node, to_node)
        if distance > self.step_size:
            dx, dy = dx / distance * self.step_size, dy / distance * self.step_size
        return (int(from_x + dx), int(from_y + dy))

    def get_neighborhood(self, nodes, new_node):
        return [node for node in nodes if self.distance(node, new_node) <= self.radius]

    def heuristic(self, a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def distance(self, a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def reconstruct_path(self, tree, current, partial=False):
        path_part = []
        while current is not None:
            path_part.append(current)
            self.draw_callback(self.maze, current[0], current[1], "blue")
            current = tree[current]
        if partial:
            self.maze.path.extend(path_part)
        else:
            self.maze.path = path_part[::-1]


class DWASolver:
    def __init__(self, maze, draw_callback):
        self.maze = maze
        self.draw_callback = draw_callback
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (-1, -1), (1, 1), (1, -1)]
        self.weight_factor = 1.0

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def solve(self):
        start = (0, 0)
        end = (self.maze.size - 1, self.maze.size - 1)
        return self.dwa_star(start, end)

    def dwa_star(self, start, end):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {start: None}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                self.reconstruct_path(came_from, current)
                return self.maze.path

            for direction in self.directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                if not self.maze.is_valid_move(neighbor[0], neighbor[1]):
                    continue

                tentative_g_score = g_score[current] + 1

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.weight_factor * self.heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, neighbor))
                    self.maze.maze[neighbor[0]][neighbor[1]] = 2  # Mark as visited
                    self.draw_callback(self.maze, neighbor[0], neighbor[1], "green")

        return None

    def reconstruct_path(self, came_from, current):
        while current is not None:
            self.maze.path.append(current)
            self.draw_callback(self.maze, current[0], current[1], "blue")
            current = came_from[current]
        self.maze.path.reverse()


class RRTSolver:
    def __init__(self, maze, draw_callback):
        self.maze = maze
        self.draw_callback = draw_callback
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (-1, -1), (1, 1), (1, -1)]
        self.step_size = 1

    def solve(self):
        start = (0, 0)
        end = (self.maze.size - 1, self.maze.size - 1)
        return self.rrt(start, end)

    def rrt(self, start, end):
        tree = {start: None}
        nodes = [start]

        while nodes:
            current = self.random_sample()
            nearest_node = self.nearest(tree, current)

            new_node = self.steer(nearest_node, current)

            if self.maze.is_valid_move(new_node[0], new_node[1]):
                tree[new_node] = nearest_node
                nodes.append(new_node)
                self.maze.maze[new_node[0]][new_node[1]] = 2  # Mark as visited
                self.draw_callback(self.maze, new_node[0], new_node[1], "green")

                if self.heuristic(new_node, end) <= self.step_size:
                    self.reconstruct_path(tree, new_node)
                    self.reconstruct_path(tree, end, partial=True)
                    return self.maze.path

        return None

    def random_sample(self):
        return (random.randint(0, self.maze.size - 1), random.randint(0, self.maze.size - 1))

    def nearest(self, tree, sample):
        return min(tree.keys(), key=lambda node: self.heuristic(node, sample))

    def steer(self, from_node, to_node):
        from_x, from_y = from_node
        to_x, to_y = to_node
        dx, dy = to_x - from_x, to_y - from_y
        distance = self.heuristic(from_node, to_node)
        if distance > self.step_size:
            dx, dy = dx / distance * self.step_size, dy / distance * self.step_size
        return (int(from_x + dx), int(from_y + dy))

    def heuristic(self, a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def reconstruct_path(self, tree, current, partial=False):
        path_part = []
        while current is not None:
            path_part.append(current)
            self.draw_callback(self.maze, current[0], current[1], "blue")
            current = tree[current]
        if partial:
            self.maze.path.extend(path_part)
        else:
            self.maze.path = path_part[::-1]


# 采用跳点搜索算法
class JPSSolver:
    def __init__(self, maze, draw_callback):
        self.maze = maze
        self.draw_callback = draw_callback
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (-1, -1), (1, 1), (1, -1)]

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def solve(self):
        start = (0, 0)
        end = (self.maze.size - 1, self.maze.size - 1)
        return self.jps(start, end)

    def jps(self, start, end):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {start: None}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                self.reconstruct_path(came_from, current)
                return self.maze.path

            neighbors = self.get_successors(current, end)
            for neighbor in neighbors:
                tentative_g_score = g_score[current] + self.heuristic(current, neighbor)

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, neighbor))
                    self.maze.maze[neighbor[0]][neighbor[1]] = 2  # Mark as visited
                    self.draw_callback(self.maze, neighbor[0], neighbor[1], "green")

        return None

    def get_successors(self, node, end):
        successors = []
        for direction in self.directions:
            next_jump = self.jump(node, direction, end)
            if next_jump:
                successors.append(next_jump)
        return successors

    def jump(self, node, direction, end):
        x, y = node
        dx, dy = direction
        nx, ny = x + dx, y + dy

        if not self.maze.is_valid_move(nx, ny):
            return None

        if (nx, ny) == end:
            return nx, ny

        if any(self.maze.is_valid_move(nx + dx, ny + dy) for dx, dy in self.directions):
            return nx, ny

        if dx != 0 and dy != 0:
            if self.jump((x + dx, y), (dx, 0), end) or self.jump((x, y + dy), (0, dy), end):
                return nx, ny
        elif dx != 0:
            if self.jump((x + dx, y), (dx, 0), end):
                return nx, ny
        elif dy != 0:
            if self.jump((x, y + dy), (0, dy), end):
                return nx, ny
        return self.jump((nx, ny), direction, end)

    def reconstruct_path(self, came_from, current):
        while current is not None:
            self.maze.path.append(current)
            self.draw_callback(self.maze, current[0], current[1], "blue")
            current = came_from[current]
        self.maze.path.reverse()


# 采用迭代加深搜索算法
class IDDFSSolver:
    def __init__(self, maze, draw_callback):
        self.maze = maze
        self.draw_callback = draw_callback
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (-1, -1), (1, 1), (1, -1)]

    def solve(self):
        start = (0, 0)
        end = (self.maze.size - 1, self.maze.size - 1)
        for depth in range(0, self.maze.size * self.maze.size):
            path = self.dfs_limited(start, end, depth)
            if path:
                return path
        return None

    def dfs_limited(self, start, end, limit):
        stack = [(start, 0)]
        came_from = {start: None}

        while stack:
            current, depth = stack.pop()

            if current == end:
                self.reconstruct_path(came_from, current)
                return self.maze.path

            if depth < limit:
                for direction in self.directions:
                    neighbor = (current[0] + direction[0], current[1] + direction[1])

                    if self.maze.is_valid_move(neighbor[0], neighbor[1]) and neighbor not in came_from:
                        stack.append((neighbor, depth + 1))
                        came_from[neighbor] = current
                        self.maze.maze[neighbor[0]][neighbor[1]] = 2  # Mark as visited
                        self.draw_callback(self.maze, neighbor[0], neighbor[1], "green")
        return None

    def reconstruct_path(self, came_from, current):
        while current is not None:
            self.maze.path.append(current)
            self.draw_callback(self.maze, current[0], current[1], "blue")
            current = came_from[current]
        self.maze.path.reverse()


# 采用Dijkstra算法
class DijkstraSolver:
    def __init__(self, maze, draw_callback):
        self.maze = maze
        self.draw_callback = draw_callback
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (-1, -1), (1, 1), (1, -1)]

    def solve(self):
        start = (0, 0)
        end = (self.maze.size - 1, self.maze.size - 1)
        return self.dijkstra(start, end)

    def dijkstra(self, start, end):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {start: None}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                self.reconstruct_path(came_from, current)
                return self.maze.path

            for direction in self.directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                temp_g_score = g_score.get(current, float('inf')) + 1

                if self.maze.is_valid_move(neighbor[0], neighbor[1]) and temp_g_score < g_score.get(neighbor,
                                                                                                    float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    heapq.heappush(open_set, (g_score[neighbor], neighbor))
                    self.maze.maze[neighbor[0]][neighbor[1]] = 2  # Mark as visited
                    self.draw_callback(self.maze, neighbor[0], neighbor[1], "green")

        return None

    def reconstruct_path(self, came_from, current):
        while current is not None:
            self.maze.path.append(current)
            self.draw_callback(self.maze, current[0], current[1], "blue")
            current = came_from[current]
        self.maze.path.reverse()


# 采用贪婪算法
class GreedySolver:
    def __init__(self, maze, draw_callback):
        self.maze = maze
        self.draw_callback = draw_callback
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (-1, -1), (1, 1), (1, -1)]

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def solve(self):
        start = (0, 0)
        end = (self.maze.size - 1, self.maze.size - 1)
        return self.greedy(start, end)

    def greedy(self, start, end):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {start: None}
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == end:
                self.reconstruct_path(came_from, current)
                return self.maze.path

            for direction in self.directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                if self.maze.is_valid_move(neighbor[0], neighbor[1]) and neighbor not in came_from:
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (self.heuristic(neighbor, end), neighbor))
                    self.maze.maze[neighbor[0]][neighbor[1]] = 2  # Mark as visited
                    self.draw_callback(self.maze, neighbor[0], neighbor[1], "green")

        return None

    def reconstruct_path(self, came_from, current):
        while current is not None:
            self.maze.path.append(current)
            self.draw_callback(self.maze, current[0], current[1], "blue")
            current = came_from[current]
        self.maze.path.reverse()


# 采用双向BFS算法
class BidirectionalSolver:
    def __init__(self, maze, draw_callback):
        self.maze = maze
        self.draw_callback = draw_callback
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (-1, -1), (1, 1), (1, -1)]

    def solve(self):
        start = (0, 0)
        end = (self.maze.size - 1, self.maze.size - 1)
        return self.bidirectional(start, end)

    def bidirectional(self, start, end):
        front_queue = [start]
        back_queue = [end]
        front_came_from = {start: None}
        back_came_from = {end: None}

        while front_queue and back_queue:
            if self.expand_front(front_queue, front_came_from, back_came_from, end):
                self.reconstruct_path(front_came_from, back_came_from, start, end)
                return self.maze.path
            if self.expand_back(back_queue, back_came_from, front_came_from, start):
                self.reconstruct_path(front_came_from, back_came_from, start, end)
                return self.maze.path

        return None

    def expand_front(self, queue, came_from, other_came_from, end):
        current = queue.pop(0)
        for direction in self.directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if self.maze.is_valid_move(neighbor[0], neighbor[1]) and neighbor not in came_from:
                came_from[neighbor] = current
                if neighbor in other_came_from:
                    return True
                queue.append(neighbor)
                self.maze.maze[neighbor[0]][neighbor[1]] = 2  # Mark as visited
                self.draw_callback(self.maze, neighbor[0], neighbor[1], "green")
        return False

    def expand_back(self, queue, came_from, other_came_from, start):
        current = queue.pop(0)
        for direction in self.directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if self.maze.is_valid_move(neighbor[0], neighbor[1]) and neighbor not in came_from:
                came_from[neighbor] = current
                if neighbor in other_came_from:
                    return True
                queue.append(neighbor)
                self.maze.maze[neighbor[0]][neighbor[1]] = 2  # Mark as visited
                self.draw_callback(self.maze, neighbor[0], neighbor[1], "green")
        return False

    def reconstruct_path(self, front_came_from, back_came_from, start, end):
        meet_point = None
        for point in front_came_from.keys():
            if point in back_came_from:
                meet_point = point
                break

        path = []
        current = meet_point
        while current is not None:
            path.append(current)
            current = front_came_from[current]
        path.reverse()
        current = back_came_from[meet_point]
        while current is not None:
            path.append(current)
            current = back_came_from[current]

        self.maze.path = path
        for (x, y) in path:
            self.draw_callback(self.maze, x, y, "blue")


# 采用DFS算法
class DFSSolver:
    def __init__(self, maze, draw_callback):
        self.maze = maze
        self.draw_callback = draw_callback
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (-1, -1), (1, 1), (1, -1)]

    def solve(self):
        start = (0, 0)
        end = (self.maze.size - 1, self.maze.size - 1)
        return self.dfs(start, end)

    def dfs(self, start, end):
        stack = [start]
        came_from = {start: None}

        while stack:
            current = stack.pop()

            if current == end:
                self.reconstruct_path(came_from, current)
                return self.maze.path

            for direction in self.directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])

                if self.maze.is_valid_move(neighbor[0], neighbor[1]) and neighbor not in came_from:
                    stack.append(neighbor)
                    came_from[neighbor] = current
                    self.maze.maze[neighbor[0]][neighbor[1]] = 2  # Mark as visited
                    self.draw_callback(self.maze, neighbor[0], neighbor[1], "green")

        return None

    def reconstruct_path(self, came_from, current):
        while current is not None:
            self.maze.path.append(current)
            self.draw_callback(self.maze, current[0], current[1], "blue")
            current = came_from[current]
        self.maze.path.reverse()


# 采用BFS算法
class BFSSolver:
    def __init__(self, maze, draw_callback):
        self.maze = maze
        self.draw_callback = draw_callback
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (-1, -1), (1, 1), (1, -1)]

    def solve(self):
        start = (0, 0)
        end = (self.maze.size - 1, self.maze.size - 1)
        return self.bfs(start, end)

    def bfs(self, start, end):
        queue = [start]
        came_from = {start: None}

        while queue:
            current = queue.pop(0)

            if current == end:
                self.reconstruct_path(came_from, current)
                return self.maze.path

            for direction in self.directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])

                if self.maze.is_valid_move(neighbor[0], neighbor[1]) and neighbor not in came_from:
                    queue.append(neighbor)
                    came_from[neighbor] = current
                    self.maze.maze[neighbor[0]][neighbor[1]] = 2  # Mark as visited
                    self.draw_callback(self.maze, neighbor[0], neighbor[1], "green")

        return None

    def reconstruct_path(self, came_from, current):
        while current is not None:
            self.maze.path.append(current)
            self.draw_callback(self.maze, current[0], current[1], "blue")
            current = came_from[current]
        self.maze.path.reverse()


# 采用A*算法
class AStarSolver:
    def __init__(self, maze, draw_callback):
        self.maze = maze
        self.draw_callback = draw_callback
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (-1, -1), (1, 1), (1, -1)]

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def solve(self):
        start = (0, 0)
        end = (self.maze.size - 1, self.maze.size - 1)
        return self.a_star(start, end)

    def a_star(self, start, end):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {start: None}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, end)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                self.reconstruct_path(came_from, current)
                return self.maze.path

            for direction in self.directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                temp_g_score = g_score.get(current, float('inf')) + 1

                if self.maze.is_valid_move(neighbor[0], neighbor[1]) and temp_g_score < g_score.get(neighbor,
                                                                                                    float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + self.heuristic(neighbor, end)

                    if neighbor not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        self.maze.maze[neighbor[0]][neighbor[1]] = 2  # Mark as visited
                        self.draw_callback(self.maze, neighbor[0], neighbor[1], "green")

        return None

    def reconstruct_path(self, came_from, current):
        while current is not None:
            self.maze.path.append(current)
            self.draw_callback(self.maze, current[0], current[1], "blue")
            current = came_from[current]
        self.maze.path.reverse()


class MazeApp:
    def __init__(self, root, maze_size=8, obstacle_prob=0.3):
        self.root = root
        self.maze_size = maze_size
        self.obstacle_prob = obstacle_prob
        self.maze = Maze(maze_size, obstacle_prob)

        self.canvas = tk.Canvas(root, width=self.maze_size * 40, height=self.maze_size * 40, bg="white")
        self.canvas.pack(side=tk.LEFT)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        self.info_frame = tk.Frame(root)
        self.info_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.solve_a_star_button = tk.Button(self.button_frame, text="A*寻找路径",
                                             command=lambda: self.solve_maze(algorithm='A*'))
        self.solve_a_star_button.grid(row=0, column=0)

        self.solve_bfs_button = tk.Button(self.button_frame, text="BFS寻找路径",
                                          command=lambda: self.solve_maze(algorithm='BFS'))
        self.solve_bfs_button.grid(row=1, column=0)

        self.solve_dfs_button = tk.Button(self.button_frame, text="DFS寻找路径",
                                          command=lambda: self.solve_maze(algorithm='DFS'))
        self.solve_dfs_button.grid(row=2, column=0)

        self.solve_greedy_button = tk.Button(self.button_frame, text="Greedy寻找路径",
                                             command=lambda: self.solve_maze(algorithm='Greedy'))
        self.solve_greedy_button.grid(row=3, column=0)

        self.solve_bidirectional_button = tk.Button(self.button_frame, text="Bidirectional寻找路径(wrong)",
                                                    command=lambda: self.solve_maze(algorithm='Bidirectional'))
        self.solve_bidirectional_button.grid(row=4, column=0)

        self.solve_iddfs_button = tk.Button(self.button_frame, text="IDDFS寻找路径(wrong)",
                                            command=lambda: self.solve_maze(algorithm='IDDFS'))
        self.solve_iddfs_button.grid(row=5, column=0)

        self.solve_dijkstra_button = tk.Button(self.button_frame, text="Dijkstra寻找路径",
                                               command=lambda: self.solve_maze(algorithm='Dijkstra'))
        self.solve_dijkstra_button.grid(row=6, column=0)

        self.solve_astar_button = tk.Button(self.button_frame, text="跳点搜索",
                                            command=lambda: self.solve_maze(algorithm='JPSS'))
        self.solve_astar_button.grid(row=7, column=0)

        self.solve_astar_button = tk.Button(self.button_frame, text="RRT(!)",
                                            command=lambda: self.solve_maze(algorithm='RRT'))
        self.solve_astar_button.grid(row=8, column=0)

        self.solve_astar_button = tk.Button(self.button_frame, text="DWA*",
                                            command=lambda: self.solve_maze(algorithm='DWA*'))
        self.solve_astar_button.grid(row=9, column=0)

        self.solve_astar_button = tk.Button(self.button_frame, text="RRT*(!)",
                                            command=lambda: self.solve_maze(algorithm='RRT*'))
        self.solve_astar_button.grid(row=10, column=0)

        self.solve_astar_button = tk.Button(self.button_frame, text="Hybrid A*",
                                            command=lambda: self.solve_maze(algorithm='Hybrid A*'))
        self.solve_astar_button.grid(row=11, column=0)

        self.solve_astar_button = tk.Button(self.button_frame, text="Potential Field",
                                            command=lambda: self.solve_maze(algorithm='Potential Field'))
        self.solve_astar_button.grid(row=12, column=0)

        self.solve_astar_button = tk.Button(self.button_frame, text="Theta*",
                                            command=lambda: self.solve_maze(algorithm='Theta*'))
        self.solve_astar_button.grid(row=13, column=0)

        self.new_maze_button = tk.Button(self.button_frame, text="生成新的迷宫", command=self.generate_new_maze)
        self.new_maze_button.grid(row=1, column=1)

        self.custom_settings_button = tk.Button(self.button_frame, text="设置迷宫", command=self.custom_settings)
        self.custom_settings_button.grid(row=2, column=1)

        self.reset_path_button = tk.Button(self.button_frame, text="重新开始", command=self.reset_path)
        self.reset_path_button.grid(row=3, column=1)

        self.info_label = tk.Label(self.info_frame, text="算法运行结果", font=('Helvetica', 14, 'bold'))
        self.info_label.pack()

        self.time_label = tk.Label(self.info_frame, text="", font=('Helvetica', 12))
        self.time_label.pack()

        self.path_length_label = tk.Label(self.info_frame, text="", font=('Helvetica', 12))
        self.path_length_label.pack()

        self.draw_maze()

    def draw_maze(self, maze=None, x=None, y=None, color=None):
        if not maze:
            maze = self.maze

        if color is None:
            color = "green"

        self.canvas.delete("all")
        self.canvas.config(width=self.maze_size * 40, height=self.maze_size * 40)
        self.root.geometry(f"{self.maze_size * 40 + 240}x{self.maze_size * 40 + 80}")  # Update window size

        for i in range(self.maze_size):
            for j in range(self.maze_size):
                cell_color = "white"
                if maze.maze[i][j] == 1:
                    cell_color = "grey"
                elif maze.maze[i][j] == 2:
                    cell_color = "darkgrey"
                self.canvas.create_rectangle(j * 40, i * 40, (j + 1) * 40, (i + 1) * 40, fill=cell_color)

        self.canvas.create_rectangle(0, 0, 40, 40, fill="blue")  # Start point
        self.canvas.create_rectangle((self.maze_size - 1) * 40, (self.maze_size - 1) * 40, self.maze_size * 40,
                                     self.maze_size * 40, fill="red")  # End point

        if x is not None and y is not None:
            self.canvas.create_rectangle(y * 40, x * 40, (y + 1) * 40, (x + 1) * 40, fill=color)
            self.root.update_idletasks()

        if self.maze.path:
            for (path_x, path_y) in self.maze.path:
                self.canvas.create_rectangle(path_y * 40 + 10, path_x * 40 + 10, (path_y + 1) * 40 - 10,
                                             (path_x + 1) * 40 - 10, fill="blue")

    def solve_maze(self, algorithm='A*'):
        start_time = datetime.now()
        result = self.maze.solve_maze(self.draw_maze, algorithm=algorithm)
        end_time = datetime.now()
        time_taken = (end_time - start_time).total_seconds()

        if result:
            print("找到路径:", self.maze.path)
            self.update_info(algorithm, time_taken, len(self.maze.path))
        else:
            print("没有找到路径。")
            self.update_info(algorithm, time_taken, 0)

    def update_info(self, algorithm, time_taken, path_length):
        self.time_label.config(text=f"{algorithm}算法运行时间: {time_taken:.4f} 秒")
        self.path_length_label.config(text=f"{algorithm}算法找到的路径长度: {path_length}")

    def generate_new_maze(self):
        self.maze = Maze(self.maze_size, self.obstacle_prob)
        self.time_label.config(text="")
        self.path_length_label.config(text="")
        self.draw_maze()

    def custom_settings(self):
        new_size = simpledialog.askinteger("迷宫尺寸", "输入迷宫尺寸:", minvalue=5, maxvalue=50)
        new_prob = simpledialog.askfloat("障碍概率", "输入障碍概率 (0-1):", minvalue=0.0,
                                         maxvalue=1.0)
        if new_size and new_prob is not None:
            self.maze_size = new_size
            self.obstacle_prob = new_prob
            self.generate_new_maze()

    def reset_path(self):
        self.maze.path = []  # 清空路径
        for i in range(self.maze.size):
            for j in range(self.maze.size):
                if self.maze.maze[i][j] == 2:  # 重置被标记为访问过的点
                    self.maze.maze[i][j] = 0
        self.time_label.config(text="")
        self.path_length_label.config(text="")
        self.draw_maze()


if __name__ == "__main__":
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()
