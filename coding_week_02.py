import numpy as np
import random
import tkinter as tk


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
        maze[0][0] = 0
        maze[self.size - 1][self.size - 1] = 0
        return maze.tolist()

    def is_valid_move(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size and self.maze[x][y] == 0

    def solve_maze(self, strategy):
        self.path = []
        solver = strategy(self)
        return solver.solve()


class DepthFirstSearchSolver:
    def __init__(self, maze):
        self.maze = maze
        self.path = []

    def solve(self):
        if self.dfs(0, 0):
            return self.path
        else:
            return None

    def dfs(self, x, y):
        if (x, y) == (self.maze.size - 1, self.maze.size - 1):
            self.path.append((x, y))
            return True

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        self.maze.maze[x][y] = 2  # Mark as visited
        self.path.append((x, y))

        for d in directions:
            next_x, next_y = x + d[0], y + d[1]
            if self.maze.is_valid_move(next_x, next_y) and self.maze.maze[next_x][next_y] == 0:
                if self.dfs(next_x, next_y):
                    return True

        self.path.pop()
        return False


class MazeApp:
    def __init__(self, root, maze_size=8, obstacle_prob=0.3):
        self.root = root
        self.maze_size = maze_size
        self.obstacle_prob = obstacle_prob
        self.maze = Maze(maze_size, obstacle_prob)

        self.canvas = tk.Canvas(root, width=self.maze_size * 40, height=self.maze_size * 40)
        self.canvas.pack()
        self.solve_button = tk.Button(root, text="寻找路径", command=self.solve_maze)
        self.solve_button.pack()
        self.new_maze_button = tk.Button(root, text="生成新迷宫", command=self.generate_new_maze)
        self.new_maze_button.pack()

        self.draw_maze()

    def draw_maze(self, path=None):
        self.canvas.delete("all")
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                color = "white" if self.maze.maze[i][j] == 0 else "grey"
                self.canvas.create_rectangle(j * 40, i * 40, (j + 1) * 40, (i + 1) * 40, fill=color)

        if path:
            for (x, y) in path:
                self.canvas.create_rectangle(y * 40 + 10, x * 40 + 10, (y + 1) * 40 - 10, (x + 1) * 40 - 10, fill="blue")

    def solve_maze(self):
        solver = DepthFirstSearchSolver(self.maze)
        path = self.maze.solve_maze(DepthFirstSearchSolver)

        if path:
            print("找到路径:", path)
            self.draw_maze(path)
        else:
            print("没找到路径")

    def generate_new_maze(self):
        self.maze = Maze(self.maze_size, self.obstacle_prob)
        self.draw_maze()


if __name__ == "__main__":
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()