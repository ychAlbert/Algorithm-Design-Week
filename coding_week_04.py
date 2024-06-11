import numpy as np
import random
import tkinter as tk
from datetime import datetime
from tkinter import simpledialog, filedialog
from PIL import ImageGrab


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
        self.maze.maze[x][y] = 2  # Mark as visited (Previously path)
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
        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        self.solve_button = tk.Button(self.button_frame, text="寻找路径", command=self.solve_maze)
        self.solve_button.grid(row=0, column=0)

        self.new_maze_button = tk.Button(self.button_frame, text="生成新迷宫", command=self.generate_new_maze)
        self.new_maze_button.grid(row=0, column=1)

        self.custom_settings_button = tk.Button(self.button_frame, text="设置迷宫参数", command=self.custom_settings)
        self.custom_settings_button.grid(row=0, column=2)

        self.save_button = tk.Button(self.button_frame, text="保存迷宫", command=self.save_maze)
        self.save_button.grid(row=0, column=3)

        self.reset_path_button = tk.Button(self.button_frame, text="重新设置路径", command=self.reset_path)
        self.reset_path_button.grid(row=0, column=4)

        self.info_label = tk.Label(root, text="")
        self.info_label.pack()

        self.draw_maze()

    def draw_maze(self, path=None):
        self.canvas.delete("all")
        self.canvas.config(width=self.maze_size * 40, height=self.maze_size * 40)
        self.root.geometry(f"{self.maze_size * 40 + 40}x{self.maze_size * 40 + 80}")  # Update window size

        for i in range(self.maze_size):
            for j in range(self.maze_size):
                color = "white"
                if self.maze.maze[i][j] == 1:
                    color = "grey"
                elif self.maze.maze[i][j] == 2:
                    color = "light grey"
                self.canvas.create_rectangle(j * 40, i * 40, (j + 1) * 40, (i + 1) * 40, fill=color)

        if path:
            for (x, y) in path:
                self.canvas.create_rectangle(y * 40 + 10, x * 40 + 10, (y + 1) * 40 - 10, (x + 1) * 40 - 10,
                                             fill="blue")

    def solve_maze(self):
        start_time = datetime.now()
        solver = DepthFirstSearchSolver(self.maze)
        path = self.maze.solve_maze(DepthFirstSearchSolver)
        end_time = datetime.now()
        time_taken = (end_time - start_time).total_seconds()

        if path:
            print("Path found:", path)
            self.draw_maze(path)
            self.info_label.config(text=f"Path found with length {len(path)} in {time_taken:.4f} seconds.")
        else:
            print("No path found")
            self.info_label.config(text="No path found.")

    def generate_new_maze(self):
        self.maze = Maze(self.maze_size, self.obstacle_prob)
        self.info_label.config(text="")
        self.draw_maze()

    def custom_settings(self):
        new_size = simpledialog.askinteger("Maze Size", "Enter maze size:", minvalue=5, maxvalue=50)
        new_prob = simpledialog.askfloat("Obstacle Probability", "Enter obstacle probability (0-1):", minvalue=0.0,
                                         maxvalue=1.0)
        if new_size and new_prob:
            self.maze_size = new_size
            self.obstacle_prob = new_prob
            self.generate_new_maze()

    def save_maze(self):
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        filename = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if filename:
            ImageGrab.grab().crop((x, y, x1, y1)).save(filename)

    def reset_path(self):
        self.info_label.config(text="")
        self.draw_maze()


if __name__ == "__main__":
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()