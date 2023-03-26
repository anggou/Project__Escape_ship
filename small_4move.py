import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
import random

H_UNIT = 7 * 5  # 픽셀 수
HEIGHT = 22  # 그리드 세로
W_UNIT = 15 * 3  # 픽셀 수
WIDTH = 30  # 그리드 가로
FLOOR = 2
PhotoImage = ImageTk.PhotoImage
AG = [1, 8, 27]  # z,y,x
lifeboat_location_1 = [1, 3, 9]
lifeboat_location_2 = [1, 18, 9]
flag_location_1_1 = [1, 3, 24]
flag_location_1_2 = [1, 18, 24]
flag_location_2_1 = [1, 4, 12]
flag_location_2_2 = [1, 17, 12]
# flag_location_3_1 = [1, 3, 9]
# flag_location_3_2 = [1, 18, 9]
fire_location = [1, 4, 9]
lifeboat_reward = 10
fire_reward = -10
block_reward = -1
stair_reward = 0
right_reward = 4
wrong_reward = -1
flag_reward_1_1 = 1
flag_reward_1_2 = 1
flag_reward_2_1 = 2
flag_reward_2_2 = 2
# flag_reward_3_1 = 3
# flag_reward_3_2 = 3

maze = [
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1],
        [1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
]


def five(n):
    if n % 5 == 0:
        a = 1
    else:
        a = 0
    return a


def place_tk(target):  # 3D  > 2D 중앙
    z = target[0]
    y = target[1]
    x = target[2]
    tk_x = ((z // 5) - five(z)) * WIDTH * W_UNIT + x * W_UNIT + W_UNIT / 2
    tk_y = ((z - 1) % 5) * (H_UNIT * HEIGHT) + y * H_UNIT + H_UNIT / 2

    return tk_x, tk_y


class Env(tk.Tk):
    def __init__(self, render_speed):
        super(Env, self).__init__()
        self.render_speed = render_speed
        self.action_space = ['f', 'b', 'l', 'r']
        self.action_size = len(self.action_space)
        self.title('DQN_4move')
        self.geometry('{0}x{1}'.format(3 * WIDTH * W_UNIT, 5 * HEIGHT * H_UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.counter = 0
        self.rewards = []
        self.goal = []
        self.explore = 0
        # 목표 지점 설정
        for z in range(FLOOR):
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    if maze[z][y][x] == 1:
                        self.set_reward([z, y, x], block_reward)

        self.rectangle = self.canvas.create_image(place_tk(AG)[0], place_tk(AG)[1],
                                                  image=self.shapes[0])  # 3D>2D
        self.set_reward(lifeboat_location_1, lifeboat_reward)
        self.set_reward(lifeboat_location_2, lifeboat_reward)
        self.set_reward(fire_location, fire_reward)
        self.set_reward(flag_location_1_1, flag_reward_1_1)
        self.set_reward(flag_location_1_2, flag_reward_1_2)
        self.set_reward(flag_location_2_1, flag_reward_2_1)
        self.set_reward(flag_location_2_2, flag_reward_2_2)
        # self.set_reward(flag_location_3_1, flag_reward_3_1)
        # self.set_reward(flag_location_3_2, flag_reward_3_2)
        self.create_widgets()
        # 메인 격자
        for col in range(0, 1 * WIDTH * W_UNIT, W_UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = col, 0, col, HEIGHT * H_UNIT * 3
            self.canvas.create_line(x0, y0, x1, y1)
        for row in range(0, 3 * HEIGHT * H_UNIT, H_UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, row, WIDTH * W_UNIT * 1, row
            self.canvas.create_line(x0, y0, x1, y1)
        # 층별 구분
        for col in range(0, 1 * WIDTH * W_UNIT, W_UNIT * WIDTH):  # 0~400 by 80
            x0, y0, x1, y1 = col, 0, col, HEIGHT * H_UNIT * 3
            self.canvas.create_line(x0, y0, x1, y1, fill='red')
        for row in range(0, 3 * HEIGHT * H_UNIT, H_UNIT * HEIGHT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, row, WIDTH * W_UNIT * 1, row
            self.canvas.create_line(x0, y0, x1, y1, fill='red')

        # # 메인 격자
        # for col in range(0, 1 * WIDTH * W_UNIT, W_UNIT):  # 0~400 by 80
        #     x0, y0, x1, y1 = col, 0, col, HEIGHT * H_UNIT * 3
        #     self.canvas.create_line(x0, y0, x1, y1)
        # for row in range(0, 3 * HEIGHT * H_UNIT, H_UNIT):  # 0~400 by 80
        #     x0, y0, x1, y1 = 0, row, WIDTH * W_UNIT * 1, row
        #     self.canvas.create_line(x0, y0, x1, y1)
        # # 층별 구분
        # for col in range(0, 1 * WIDTH * W_UNIT, W_UNIT * WIDTH):  # 0~400 by 80
        #     x0, y0, x1, y1 = col, 0, col, HEIGHT * H_UNIT * 3
        #     self.canvas.create_line(x0, y0, x1, y1, fill='red')
        # for row in range(0, 3 * HEIGHT * H_UNIT, H_UNIT * HEIGHT):  # 0~400 by 80
        #     x0, y0, x1, y1 = 0, row, WIDTH * W_UNIT * 1, row
        #     self.canvas.create_line(x0, y0, x1, y1, fill='red')

    def create_widgets(self):
        self.label = tk.Label(self, text="Render(below : 0.1) : ")
        self.label.place(x=WIDTH * W_UNIT + 10, y=10)

        self.entry = tk.Entry(self)
        self.entry.place(x=WIDTH * W_UNIT + 10, y=30)

        self.button_1 = tk.Button(self, text="Submit", command=self.submit)
        self.button_1.place(x=WIDTH * W_UNIT + 10, y=50)

        self.button_2 = tk.Button(self, text="Explore", command=self.set_explore, relief=tk.RAISED)
        self.button_2.place(x=WIDTH * W_UNIT + 10, y=100)
        self.button_2.bind('<ButtonPress-1>', self.on_press)
        self.button_2.bind_all('<ButtonRelease-1>', self.on_release)

    def set_explore(self):
        self.explore = 1

    def on_press(self, event):
        print("Button pressed")
        self.button_2.config(relief=tk.SUNKEN)
        self.set_explore()

    def on_release(self, event):
        print("Button released")
        self.button_2.config(relief=tk.RAISED)
        self.explore = 0

    def draw_from_policy(self, state, q_values):  # state, q함수
        font = 7
        dz = state[0][0]  # re-ag
        dy = state[0][1]
        dx = state[0][2]
        z = lifeboat_location_1[0] - dz
        y = lifeboat_location_1[1] - dy
        x = lifeboat_location_1[2] - dx
        self.canvas.create_text(x * W_UNIT + W_UNIT / 2 + W_UNIT / 4,
                                (z - 1) * HEIGHT * H_UNIT + y * H_UNIT + H_UNIT / 2 - H_UNIT / 4,
                                fill="black", text=round(q_values[0][0].numpy(), 1), font=('Helvetica', font),
                                anchor="nw", tags="text")
        self.canvas.create_text(x * W_UNIT + W_UNIT / 2 - W_UNIT / 2,
                                (z - 1) * HEIGHT * H_UNIT + y * H_UNIT + H_UNIT / 2 - H_UNIT / 4,
                                fill="black", text=round(q_values[0][1].numpy(), 1), font=('Helvetica', font),
                                anchor="nw", tags="text")
        self.canvas.create_text(x * W_UNIT + W_UNIT / 3,
                                (z - 1) * HEIGHT * H_UNIT + y * H_UNIT + H_UNIT / 2 - H_UNIT / 4 - H_UNIT / 4,
                                fill="black", text=round(q_values[0][2].numpy(), 1), font=('Helvetica', font),
                                anchor="nw", tags="text")
        self.canvas.create_text(x * W_UNIT + W_UNIT / 3,
                                (z - 1) * HEIGHT * H_UNIT + y * H_UNIT + H_UNIT / 2 + H_UNIT / 3,
                                fill="black", text=round(q_values[0][3].numpy(), 1), font=('Helvetica', font),
                                anchor="nw", tags="text")
        print(q_values[0][0].numpy(), q_values[0][1].numpy(), q_values[0][2].numpy(), q_values[0][3].numpy())

    def _build_canvas(self):
        canvas = tk.Canvas(self, width=3 * WIDTH * W_UNIT, height=5 * HEIGHT * H_UNIT)

        for z in range(FLOOR):
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    if maze[z][y][x] == 0:
                        for i in range(1, 6):
                            if z == i:
                                canvas.create_image(x * W_UNIT + W_UNIT / 2,
                                                    (z - 1) * HEIGHT * H_UNIT + y * H_UNIT + H_UNIT / 2,
                                                    image=self.shapes[6])
                        for i in range(6, 11):
                            if z == i:
                                canvas.create_image(WIDTH * W_UNIT + x * W_UNIT + W_UNIT / 2,
                                                    (z - 6) * HEIGHT * H_UNIT + y * H_UNIT + H_UNIT / 2,
                                                    image=self.shapes[6])
                        for i in range(11, FLOOR - 1):
                            if z == i:
                                canvas.create_image(2 * WIDTH * W_UNIT + x * W_UNIT + W_UNIT / 2,
                                                    (z - 11) * HEIGHT * H_UNIT + y * H_UNIT + H_UNIT / 2,
                                                    image=self.shapes[6])
        canvas.pack()

        return canvas

    def load_images(self):
        rectangle = PhotoImage(Image.open("../Escape_ship/img/rectangle.png").resize((W_UNIT, H_UNIT)))
        triangle = PhotoImage(Image.open("../Escape_ship/img/triangle.png").resize((W_UNIT, H_UNIT)))
        yellow = PhotoImage(Image.open("../Escape_ship/img/yellow.png").resize((W_UNIT, H_UNIT)))
        FIRE = PhotoImage(Image.open("../Escape_ship/img/FIRE.png").resize((W_UNIT, H_UNIT)))
        lifeboat = PhotoImage(Image.open("../Escape_ship/img/lifeboat.png").resize((W_UNIT, H_UNIT)))
        block = PhotoImage(Image.open("../Escape_ship/img/block.png").resize((W_UNIT, H_UNIT)))
        pink = PhotoImage(Image.open("../Escape_ship/img/pink.png").resize((W_UNIT, H_UNIT)))
        up = PhotoImage(Image.open("../Escape_ship/img/up.png").resize((W_UNIT, H_UNIT)))
        down = PhotoImage(Image.open("../Escape_ship/img/down.png").resize((W_UNIT, H_UNIT)))
        return rectangle, triangle, yellow, FIRE, lifeboat, block, pink, up, down

    def reset_reward(self):

        for reward in self.rewards:
            self.canvas.delete(reward['figure'])
        self.rewards.clear()
        self.goal.clear()
        self.set_reward(lifeboat_location_1, lifeboat_reward)
        self.set_reward(lifeboat_location_2, lifeboat_reward)
        self.set_reward(fire_location, fire_reward)
        self.set_reward(flag_location_1_1, flag_reward_1_1)
        self.set_reward(flag_location_1_2, flag_reward_1_2)
        self.set_reward(flag_location_2_1, flag_reward_2_1)
        self.set_reward(flag_location_2_2, flag_reward_2_2)
        # self.set_reward(flag_location_3_1, flag_reward_3_1)
        # self.set_reward(flag_location_3_2, flag_reward_3_2)
        for z in range(FLOOR):
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    if maze[z][y][x] == 1:
                        self.set_reward([z, y, x], block_reward)

    def submit(self):
        number = self.entry.get()
        print("Number entered:", number)
        try:
            value = float(number)
            self.render_speed = value
            print("Render speed set to:", self.render_speed)
        except ValueError:
            print("Invalid number entered")

    def set_reward(self, state, reward):  # 들어가는 state는 3D

        temp = {}
        tk_x = place_tk(state)[0]
        tk_y = place_tk(state)[1]

        if reward == lifeboat_reward:
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image(tk_x, tk_y, image=self.shapes[4])
            # print(reward, tk_x, tk_y)
            self.goal.append(temp['figure'])
        # 보상이 1이면 출력 4개
        elif reward == fire_reward:  # -1
            # temp['direction'] = -1
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image(tk_x, tk_y, image=self.shapes[3])
        elif reward == block_reward:  # -1
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image(tk_x, tk_y, image=self.shapes[5])
        elif reward == right_reward:  # -1
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image(tk_x, tk_y, image=self.shapes[2])
        elif reward == wrong_reward:  # -1
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image(tk_x, tk_y, image=self.shapes[2])
        elif reward == stair_reward:  # -1
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image(tk_x, tk_y, image=self.shapes[2])
        elif reward == flag_reward_1_1:  # -1
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image(tk_x, tk_y, image=self.shapes[6])
        elif reward == flag_reward_1_2:  # -1
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image(tk_x, tk_y, image=self.shapes[6])
        elif reward == flag_reward_2_1:  # -1
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image(tk_x, tk_y, image=self.shapes[6])
        elif reward == flag_reward_2_2:  # -1
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image(tk_x, tk_y, image=self.shapes[6])
        # elif reward == flag_reward_3_1:  # -1
        #     temp['reward'] = reward
        #     temp['figure'] = self.canvas.create_image(tk_x, tk_y, image=self.shapes[6])
        # elif reward == flag_reward_3_2:  # -1
        #     temp['reward'] = reward
        #     temp['figure'] = self.canvas.create_image(tk_x, tk_y, image=self.shapes[6])
        temp['coords'] = self.canvas.coords(temp['figure'])
        temp['state'] = state  # 입력을 그대로 3d
        self.rewards.append(temp)

    # goal에 도착했는지 확인
    def check_if_reward(self, state):
        # print(state)
        check_list = {}
        check_list['if_goal'] = False
        rewards = 0
        for reward in self.rewards:
            # print(reward['state'])
            if reward['state'] == state:
                rewards += reward['reward']
                # print('reward={0},state={1}'.format(rewards, state))
                if reward['reward'] == lifeboat_reward:
                    check_list['if_goal'] = True

        check_list['rewards'] = rewards
        return check_list

    def reset(self):
        self.update()
        time.sleep(0.5)
        x, y = self.canvas.coords(self.rectangle)  # 2D, 가운데
        # 이동한 만큼 다시 원점으로 되돌리기
        self.canvas.move(self.rectangle,
                         place_tk(AG)[0] - x,
                         place_tk(AG)[1] - y)
        self.reset_reward()
        return self.get_state(self.rectangle)  # 12개 출력(장애물 4, AG 4개) 좌표는 3D

    # not done (미도착) 이면 step 할때마다 다음 위치, 보상, 도착여부(done) 확인
    def step(self, action):  # 입력 6개 중에 하나
        self.counter += 1
        self.render()

        # if self.counter % 2 == 1:
        #     self.rewards = self.move_rewards()  # 에이전트 의 위치에 따라 , 보상을 위치를 변경 , 해당 층으로

        next_state = self.move(self.rectangle, action)  # 3D 로 출력 s'
        # print(next_coords)
        check = self.check_if_reward(next_state)  # reward, goal dict
        done = check['if_goal']
        reward = check['rewards']
        self.canvas.tag_raise(self.rectangle)  # 눈에 띄게 해줌

        state_all = self.get_state(self.rectangle)
        print("move_{} next_state_{} reward_{}".format(action, next_state, reward))
        return state_all, reward, done

    def get_state(self, Agent):

        location = self.find_rectangle(Agent)  # rec > 3D

        states = list()

        for reward in self.rewards:  # 총8개 (장애물 1개 * 4, 목표 1개 * 4)
            # states.append(reward['direction'])
            if reward['reward'] == lifeboat_reward:
                reward_location = reward['state']
                states.append(reward_location[0] - location[0])  # z상대거리
                states.append(reward_location[1] - location[1])  # y상대거리
                states.append(reward_location[2] - location[2])  # x상대거리
                states.append(1)
            elif reward['reward'] == fire_reward:
                reward_location = reward['state']
                states.append(reward_location[0] - location[0])  # 상대거리
                states.append(reward_location[1] - location[1])  # 상대거리
                states.append(reward_location[2] - location[2])  # 상대거리
                states.append(-1)
            else:
                pass

        return states

    def move_rewards(self):
        new_rewards = []
        for temp in self.rewards:
            if temp['reward'] == lifeboat_reward:
                new_rewards.append(temp)
                continue
            temp['coords'] = self.canvas.coords(temp['figure'])  # 불 움직임
            temp['state'] = self.find_rectangle(temp['figure'])  # 2D>3D
            new_rewards.append(temp)
        return new_rewards

    # rewards중에 라이프보트는 안움직이고, reward 값을 그대로 넣고 불 의경우에는 move_const(불움직임) 한 위치 와

    def move_const(self, target):
        x, y = self.canvas.coords(target['figure'])
        return x, y

    def find_rectangle(self, target):  # input (뚱x,뚱y), tkinter를 3차원으로
        temp = self.canvas.coords(target)  # x,y , 네모 가운데 지정
        z = 0  # 변수 초기화
        y = 0  # 변수 초기화
        x = 0  # 변수 초기화
        for j in range(0, 3):
            if temp[0] < (j + 1) * WIDTH * W_UNIT + W_UNIT / 2:
                for i in range(0, 5):
                    if temp[1] < (i + 1) * HEIGHT * H_UNIT + H_UNIT / 2:
                        z = i + 1 + 5 * j
                        y = (temp[1] - i * HEIGHT * H_UNIT - H_UNIT / 2) / H_UNIT
                        x = (temp[0] - j * WIDTH * W_UNIT - W_UNIT / 2) / W_UNIT
                        break
                break
        return [int(z), int(y), int(x)]  # maze  좌표

    def move(self, target, action):  # target=rectangle , tkinter

        base_action = np.array([0, 0])  # 앞뒤 , 좌우 , Tkinter
        location = self.find_rectangle(target)  # tkinter를 3차원으로
        # print(location)
        if action == 0 and location[2] < WIDTH - 1 and maze[location[0]][location[1]][location[2] + 1] != 1:
            base_action[0] += W_UNIT  # 앞
        elif action == 1 and location[2] > 0 and maze[location[0]][location[1]][location[2] - 1] != 1:
            base_action[0] -= W_UNIT  # 뒤
        elif action == 2 and location[1] > 0 and maze[location[0]][location[1] - 1][location[2]] != 1:
            base_action[1] -= H_UNIT  # 좌
        elif action == 3 and location[1] < HEIGHT - 1 and maze[location[0]][location[1] + 1][location[2]] != 1:
            base_action[1] += H_UNIT  # 우
        else:
            base_action[0] = 0
            base_action[1] = 0

        self.canvas.move(target, base_action[0], base_action[1])  # tkinter 움직임
        s_ = self.find_rectangle(target)  # 3D
        return s_

    def render(self):
        # 게임 속도 조정
        time.sleep(self.render_speed)
        self.update()
