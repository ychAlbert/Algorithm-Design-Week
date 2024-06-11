import numpy as np
import random
import tkinter as tk
from datetime import datetime
from tkinter import simpledialog, filedialog
from PIL import ImageGrab
from collections import deque


# **广度优先搜索 (BFS)**：
# 在 BreadthFirstSearchSolver 类中实现 BFS，该算法通过使用队列按层次遍历节点，保证找到的路径是最短路径。
# 使用 came_from 字典来记录每个节点的前置节点，以便在找到目标节点后回溯路径。
# 动态绘制路径：
# 使用 self.draw_callback 回调函数，在 BFS 遍历过程中动态更新画布 (self.root.update_idletasks())。
# 在迷宫找到路径后，通过 reconstruct_path 函数将最终路径标记为蓝色。

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
        maze[0][0] = 0  # Start point
        maze[self.size - 1][self.size - 1] = 0  # End point
        return maze.tolist()

    def is_valid_move(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size and self.maze[x][y] == 0

    def solve_maze(self, draw_callback):
        self.path = []
        solver = BreadthFirstSearchSolver(self, draw_callback)
        result = solver.solve()
        return result


class BreadthFirstSearchSolver:
    def __init__(self, maze, draw_callback):
        self.maze = maze
        self.draw_callback = draw_callback
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (-1, -1), (1, 1), (1, -1)]

    def solve(self):
        start = (0, 0)
        end = (self.maze.size - 1, self.maze.size - 1)
        return self.bfs(start, end)

    def bfs(self, start, end):
        queue = deque([start])
        came_from = {start: None}
        self.maze.maze[start[0]][start[1]] = 2

        while queue:
            current = queue.popleft()

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


class MazeApp:
    def __init__(self, root, maze_size=8, obstacle_prob=0.3):
        self.root = root
        self.maze_size = maze_size
        self.obstacle_prob = obstacle_prob
        self.maze = Maze(maze_size, obstacle_prob)

        self.canvas = tk.Canvas(root, width=self.maze_size * 40, height=self.maze_size * 40, bg="white")
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

        self.reset_path_button = tk.Button(self.button_frame, text="重新寻找路径", command=self.reset_path)
        self.reset_path_button.grid(row=0, column=4)

        self.info_label = tk.Label(root, text="")
        self.info_label.pack()

        self.draw_maze()

    def draw_maze(self, maze=None, x=None, y=None, color=None):
        if not maze:
            maze = self.maze

        if color is None:
            color = "green"

        self.canvas.delete("all")
        self.canvas.config(width=self.maze_size * 40, height=self.maze_size * 40)
        self.root.geometry(f"{self.maze_size * 40 + 40}x{self.maze_size * 40 + 80}")  # Update window size

        for i in range(self.maze_size):
            for j in range(self.maze_size):
                cell_color = "white"
                if maze.maze[i][j] == 1:
                    cell_color = "grey"
                elif maze.maze[i][j] == 2:
                    cell_color = "dark grey"
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

    def solve_maze(self):
        start_time = datetime.now()
        result = self.maze.solve_maze(self.draw_maze)
        end_time = datetime.now()
        time_taken = (end_time - start_time).total_seconds()

        if result:
            print("Path found:", self.maze.path)
            self.info_label.config(text=f"Path found with length {len(self.maze.path)} in {time_taken:.4f} seconds.")
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
