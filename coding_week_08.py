import numpy as np
import random
import tkinter as tk
from datetime import datetime
from tkinter import simpledialog, filedialog
from PIL import ImageGrab
from collections import defaultdict
import math

#蒙特卡罗树搜索（MCTS）：
#主要步骤：选择、扩展、模拟和反向传播。
#：选择当前最佳节点进行扩展。
#扩展：扩展节点以生成可能的移动。
#：进行随机模拟以评估扩展的节点。
#反向传播：根据模拟结果更新访问次数和胜利次数。
#辅助函数：
#mcts：执行MCTS算法的主函数。从当前节点开始选择并扩展，直到找到目标节点。
#expand：扩展当前节点以生成可能的移动。
#uct：计算每个可能移动的UCT（Upper Confidence Bound for Trees）值，以选择最佳移动。
#simulate：对当前节点进行随机模拟并评估路径质量。
#update：根据模拟结果更新访问次数和胜利次数。

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
        solver = MCTSSolver(self, draw_callback)
        result = solver.solve()
        return result


class MCTSSolver:
    def __init__(self, maze, draw_callback):
        self.maze = maze
        self.draw_callback = draw_callback
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1),(1, 1), (-1, -1), (1, -1), (-1, 1)]
        self.visits = defaultdict(int)
        self.wins = defaultdict(int)

    def solve(self):
        start = (0, 0)
        end = (self.maze.size - 1, self.maze.size - 1)
        node = (start, [])
        while node[0] != end:
            node = self.mcts(node)
        self.maze.path = node[1] + [end]
        self.reconstruct_path()
        return self.maze.path

    def mcts(self, node):
        state, path = node
        if self.visits[state] == 0:
            self.expand(node)
            return node

        best_move = max(self.directions, key=lambda d: self.uct(state, d))
        next_state = (state[0] + best_move[0], state[1] + best_move[1])
        if not self.maze.is_valid_move(next_state[0], next_state[1]):
            return node

        next_node = (next_state, path + [next_state])
        reward = self.simulate(next_node)
        self.update(state, reward)
        return next_node

    def expand(self, node):
        state, _ = node
        for d in self.directions:
            next_state = (state[0] + d[0], state[1] + d[1])
            if self.maze.is_valid_move(next_state[0], next_state[1]):
                self.visits[next_state] = 0
                self.wins[next_state] = 0

    def uct(self, state, direction):
        next_state = (state[0] + direction[0], state[1] + direction[1])
        if self.visits[next_state] == 0:
            return float('inf')
        return self.wins[next_state] / self.visits[next_state] + math.sqrt(2 * math.log(self.visits[state]) / self.visits[next_state])

    def simulate(self, node):
        state, path = node
        while state != (self.maze.size - 1, self.maze.size - 1) and len(path) < self.maze.size * 2:
            direction = random.choice(self.directions)
            next_state = (state[0] + direction[0], state[1] + direction[1])
            if self.maze.is_valid_move(next_state[0], next_state[1]):
                path.append(next_state)
                state = next_state
        return 1 if state == (self.maze.size - 1, self.maze.size - 1) else 0

    def update(self, state, reward):
        self.visits[state] += 1
        self.wins[state] += reward

    def reconstruct_path(self):
        for (x, y) in self.maze.path:
            self.draw_callback(self.maze, x, y, "blue")


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

        self.custom_settings_button = tk.Button(self.button_frame, text="设置参数", command=self.custom_settings)
        self.custom_settings_button.grid(row=0, column=2)

        self.save_button = tk.Button(self.button_frame, text="保存迷宫", command=self.save_maze)
        self.save_button.grid(row=0, column=3)

        self.reset_path_button = tk.Button(self.button_frame, text="重新设置路径", command=self.reset_path)
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
                    cell_color = "darkgrey"
                self.canvas.create_rectangle(j * 40, i * 40, (j + 1) * 40, (i + 1) * 40, fill=cell_color)

        self.canvas.create_rectangle(0, 0, 40, 40, fill="blue")  # Start point
        self.canvas.create_rectangle((self.maze_size - 1) * 40, (self.maze_size - 1) * 40, self.maze_size * 40, self.maze_size * 40, fill="red")  # End point

        if x is not None and y is not None:
            self.canvas.create_rectangle(y * 40, x * 40, (y + 1) * 40, (x + 1) * 40, fill=color)
            self.root.update_idletasks()

        if self.maze.path:
            for (path_x, path_y) in self.maze.path:
                self.canvas.create_rectangle(path_y * 40 + 10, path_x * 40 + 10, (path_y + 1) * 40 - 10, (path_x + 1) * 40 - 10, fill="blue")

    def solve_maze(self):
        start_time = datetime.now()
        result = self.maze.solve_maze(self.draw_maze)
        end_time = datetime.now()
        time_taken = (end_time - start_time).total_seconds()

        if result:
            print("找到路径:", self.maze.path)
            self.info_label.config(text=f"找到路径长度为 {len(self.maze.path)} 在 {time_taken:.4f} 秒.")
        else:
            print("未找到路径.")
            self.info_label.config(text="未找到路径.")

    def generate_new_maze(self):
        self.maze = Maze(self.maze_size, self.obstacle_prob)
        self.info_label.config(text="")
        self.draw_maze()

    def custom_settings(self):
        new_size = simpledialog.askinteger("地图大小", "输入地图大小:", minvalue=5, maxvalue=50)
        new_prob = simpledialog.askfloat("障碍概率", "输入障碍概率 (0-1):", minvalue=0.0, maxvalue=1.0)
        if new_size and new_prob is not None:
            self.maze_size = new_size
            self.obstacle_prob = new_prob
            self.generate_new_maze()

    def save_maze(self):
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if filename:
            ImageGrab.grab().crop((x, y, x1, y1)).save(filename)

    def reset_path(self):
        self.info_label.config(text="")
        self.draw_maze()


if __name__ == "__main__":
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()