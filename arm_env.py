#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pyglet

pyglet.clock.set_fps_limit(10000)


class ArmEnv(object):
    action_bound = [-1, 1] # Define o alcance de cada action, onde: -1 = 360 graus anti-horarios | 1 = 360 graus horários em cada eixo. 
    action_dim = 2 # Tamanho da saída na predição - numero de actions - cada eixo tem sua propria action
    state_dim = 7 # Tamanho da entrada para predição - features
    dt = .1  # refresh rate
    arm1l = 100
    arm2l = 100
    viewer = None
    viewer_xy = (400, 400)
    get_point = False
    mouse_in = np.array([False])
    point_l = 15  # Limite para calculo de recompensa
    grab_counter = 0

    def __init__(self, mode='easy'):
        # node1 (l, d_rad, x, y),
        # node2 (l, d_rad, x, y)

        self.mode = mode
        self.arm_info = np.zeros((2, 4))
        self.arm_info[0, 0] = self.arm1l
        self.arm_info[1, 0] = self.arm2l
        self.point_info = np.array([250, 303])
        self.point_info_init = self.point_info.copy()
        self.center_coord = np.array(self.viewer_xy) / 2

    def step(self, action):

        '''if action[0] == 0:
            z0 = z0 + 1.
        if action[0] == 1:
            y0 = y0 + 1.
        if action[0] == 2:
            x0 = x0 + 1.
        if action[0] == 3:
            x0 = x0 - 1.
        if action[0] == 4:
            y0 = y0 - 1.
        if action[0] == 5:
            z0 = z0 - 1.'''

        action = np.clip(action, *self.action_bound)

        self.arm_info[:, 1] += action * self.dt
        self.arm_info[:, 1] %= np.pi * 2

        arm1rad = self.arm_info[0, 1]
        arm2rad = self.arm_info[1, 1]

        arm1dx_dy = np.array([self.arm_info[0, 0] * np.cos(arm1rad), self.arm_info[0, 0] * np.sin(arm1rad)])
        arm2dx_dy = np.array([self.arm_info[1, 0] * np.cos(arm2rad), self.arm_info[1, 0] * np.sin(arm2rad)])
        self.arm_info[0, 2:4] = self.center_coord + arm1dx_dy  # (x1, y1)
        self.arm_info[1, 2:4] = self.arm_info[0, 2:4] + arm2dx_dy  # (x2, y2)

        s, arm2_distance = self._get_state()  # s = in_point + 4 coordenadas divido por 200, dois pontos para 'distancia do centro', penultima posição

        r = self._r_func(arm2_distance)

        return s, r, self.get_point # get_point diz se está no alvo ou não False-True

    def reset(self):
        self.get_point = False
        self.grab_counter = 0

        if self.mode == 'hard':
            pxy = np.clip(np.random.rand(2) * self.viewer_xy[0], 100, 300)
            self.point_info[:] = pxy


        else:
            arm1rad, arm2rad = np.random.rand(2) * np.pi * 2
            self.arm_info[0, 1] = arm1rad
            self.arm_info[1, 1] = arm2rad
            arm1dx_dy = np.array([self.arm_info[0, 0] * np.cos(arm1rad), self.arm_info[0, 0] * np.sin(arm1rad)])
            arm2dx_dy = np.array([self.arm_info[1, 0] * np.cos(arm2rad), self.arm_info[1, 0] * np.sin(arm2rad)])
            # print(arm1dx_dy)
            # print(arm2dx_dy)
            self.arm_info[0, 2:4] = self.center_coord + arm1dx_dy  # (x1, y1)
            self.arm_info[1, 2:4] = self.arm_info[0, 2:4] + arm2dx_dy  # (x2, y2)

            self.point_info[:] = self.point_info_init
        return self._get_state()[0]

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(*self.viewer_xy, self.arm_info, self.point_info, self.point_l, self.mouse_in)
        self.viewer.render()

    def sample_action(self):
        return np.random.uniform(*self.action_bound, size=self.action_dim)

    def set_fps(self, fps=30):
        pyglet.clock.set_fps_limit(fps)
        xax = 0

    def _get_state(self):
        # return the distance (dx, dy) between arm finger point with blue point
        arm_end = self.arm_info[:, 2:4]
        # arm = [[x_drone,y_drone]]
        t_arms = np.ravel(arm_end - self.point_info)
        # print(arm_end)
        # print(t_arms)
        center_dis = (self.center_coord - self.point_info) / 200  # distancia do alvo para o veiculo

        # print(self.point_info)
        in_point = 1 if self.grab_counter > 0 else 0  ## se o contador de ciclos é maior que zero o status de
                                                      ## if in_point = 1, caso contrário é zero.

        '''a, b = arm_end[0]
        c, d = arm_end[1]

        g, h = self.point_info

        # e,f,g,h = t_arms
        e, f = self.center_coord

        op = open("teste.csv", "a")
        op.write(str(a) + "," + str(b) + "," + str(c) + "," + str(d) + "," + str(e) + "," + str(
            f) + "," + str(g) + "," + str(h) + "\n")
        op.close()'''

        return np.hstack([in_point, t_arms / 200, center_dis,  ##
                          ]), t_arms[-2:]

    def _r_func(self, distance):
        t = 50  # Número de ciclos (piso) para dar início à multiplicação da recompensa por permanencia em proximidade

        ######################################################
        #                                                    #
        #     CÁLCULO DA RECOMPENSA BASEADO NA DISTÂNCIA     #
        #                                                    #
        ######################################################

        abs_distance = np.sqrt(np.sum(np.square(distance)))  ## descobrir a distância euclidiana
        r = -abs_distance / 200  # Cálculo da recompensa baseado na distância Absoluta calculada acima, quanto mais longe menor a distancia

        if abs_distance < self.point_l and (not self.get_point):  # Verifica se a distância está dentro do limite
            r += 1.  # a recompensa é somada de 1 caso esteja abaixo do limite (15)
            self.grab_counter += 1  # aumento de 1 na contagem de ciclos
            if self.grab_counter > t:  # verifica se o numero de ciclos está acima do piso (50)
                r += 10.  # Se a distancia é menor que o limite e se estiver aqui por mais de 50 ciclos
                self.get_point = True  # a recompensa é multiplicada por 10
        elif abs_distance > self.point_l:  # se a distância é maior que o limite (15)
            self.grab_counter = 0  # zera o contador de ciclos
            self.get_point = False
        return r


class Viewer(pyglet.window.Window):
    color = {
        'background': [1] * 3 + [1]
    }
    fps_display = pyglet.clock.ClockDisplay()
    bar_thc = 5

    def __init__(self, width, height, arm_info, point_info, point_l, mouse_in):
        # print(width, height, arm_info, point_info, point_l)
        super(Viewer, self).__init__(width, height, resizable=False, caption='Arm',
                                     vsync=False)  # vsync=False to not use the monitor FPS
        self.set_location(x=80, y=10)
        pyglet.gl.glClearColor(*self.color['background'])

        self.arm_info = arm_info
        self.point_info = point_info
        self.mouse_in = mouse_in
        self.point_l = point_l

        self.center_coord = np.array((min(width, height) / 2,) * 2)
        self.batch = pyglet.graphics.Batch()

        arm1_box, arm2_box, point_box = [0] * 8, [0] * 8, [0] * 8
        c1, c2, c3 = (249, 86, 86) * 4, (86, 109, 249) * 4, (249, 39, 65) * 4
        self.point = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', point_box), ('c3B', c2))
        self.arm1 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', arm1_box), ('c3B', c1))
        self.arm2 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', arm2_box), ('c3B', c1))

    def render(self):
        # pyglet.clock.tick()
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()
        self.fps_display.draw()

    def _update_arm(self):
        point_l = self.point_l
        point_box = (self.point_info[0] - point_l, self.point_info[1] - point_l,
                     self.point_info[0] + point_l, self.point_info[1] - point_l,
                     self.point_info[0] + point_l, self.point_info[1] + point_l,
                     self.point_info[0] - point_l, self.point_info[1] + point_l)
        self.point.vertices = point_box

        arm1_coord = (*self.center_coord, *(self.arm_info[0, 2:4]))  # (x0, y0, x1, y1)

        arm2_coord = (*(self.arm_info[0, 2:4]), *(self.arm_info[1, 2:4]))  # (x1, y1, x2, y2)

        arm1_thick_rad = np.pi / 2 - self.arm_info[0, 1]

        x01, y01 = arm1_coord[0] - np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[1] + np.sin(
            arm1_thick_rad) * self.bar_thc

        x02, y02 = arm1_coord[0] + np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[1] - np.sin(
            arm1_thick_rad) * self.bar_thc

        x11, y11 = arm1_coord[2] + np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[3] - np.sin(
            arm1_thick_rad) * self.bar_thc

        x12, y12 = arm1_coord[2] - np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[3] + np.sin(
            arm1_thick_rad) * self.bar_thc

        arm1_box = (x01, y01, x02, y02, x11, y11, x12, y12)

        arm2_thick_rad = np.pi / 2 - self.arm_info[1, 1]

        x11_, y11_ = arm2_coord[0] + np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[1] - np.sin(
            arm2_thick_rad) * self.bar_thc

        x12_, y12_ = arm2_coord[0] - np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[1] + np.sin(
            arm2_thick_rad) * self.bar_thc

        x21, y21 = arm2_coord[2] - np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[3] + np.sin(
            arm2_thick_rad) * self.bar_thc

        x22, y22 = arm2_coord[2] + np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[3] - np.sin(
            arm2_thick_rad) * self.bar_thc

        arm2_box = (x11_, y11_, x12_, y12_, x21, y21, x22, y22)

        self.arm1.vertices = arm1_box

        self.arm2.vertices = arm2_box

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.UP:
            self.arm_info[0, 1] += .1
            # print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.DOWN:
            self.arm_info[0, 1] -= .1
            # print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.LEFT:
            self.arm_info[1, 1] += .1
            # print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.RIGHT:
            self.arm_info[1, 1] -= .1
            # print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.Q:
            pyglet.clock.set_fps_limit(1000)
        elif symbol == pyglet.window.key.A:
            pyglet.clock.set_fps_limit(30)

    def on_mouse_motion(self, x, y, dx, dy):
        self.point_info[:] = [x, y]

    def on_mouse_enter(self, x, y):
        self.mouse_in[0] = True

    def on_mouse_leave(self, x, y):
        self.mouse_in[0] = False
