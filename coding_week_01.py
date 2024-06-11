import numpy as np
import random
import tkinter as tk


class MazeGenerator:
    @staticmethod
    def generate_maze(size, obstacle_prob=0.3):
        maze = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                if random.random() < obstacle_prob:
                    maze[i][j] = 1
        # Ensure start and end are open
        maze[0][0] = 0
        maze[size-1][size-1] = 0
        return maze.tolist()


class MazeSolverWithGUI:
    def __init__(self, maze):
        self.maze = maze
        self.rows = len(maze)
        self.cols = len(maze[0])
        self.visited = [[False]*self.cols for _ in range(self.rows)]
        self.path = []

    def is_valid_move(self, x, y):
        return 0 <= x < self.rows and 0 <= y < self.cols and self.maze[x][y] == 0 and not self.visited[x][y]

    def dfs(self, x, y):
        if x == self.rows - 1 and y == self.cols - 1:
            self.path.append((x, y))
            return True

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        self.visited[x][y] = True
        self.path.append((x, y))

        for d in directions:
            next_x, next_y = x + d[0], y + d[1]
            if self.is_valid_move(next_x, next_y) and self.dfs(next_x, next_y):
                return True

        self.path.pop()
        return False

    def solve(self):
        if self.dfs(0, 0):
            return self.path
        else:
            return None

    def draw_maze(self, path=None):
        root = tk.Tk()
        canvas = tk.Canvas(root, width=self.cols*40, height=self.rows*40)
        canvas.pack()

        for i in range(self.rows):
            for j in range(self.cols):
                color = "white" if self.maze[i][j] == 0 else "grey"
                canvas.create_rectangle(j*40, i*40, (j+1)*40, (i+1)*40, fill=color)

        if path:
            for (x, y) in path:
                canvas.create_rectangle(y*40+10, x*40+10, (y+1)*40-10, (x+1)*40-10, fill="blue")

        root.mainloop()


# Automatically generate an 8x8 maze
maze_generator = MazeGenerator()
maze = maze_generator.generate_maze(8)

# Solve and visualize the maze
solver = MazeSolverWithGUI(maze)
path = solver.solve()

if path:
    print("找到路径:", path)
else:
    print("没找到路径")

solver.draw_maze(path)