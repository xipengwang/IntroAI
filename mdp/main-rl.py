#!/usr/bin/env python3

#EECS 492(Winter 2019) Discussion
#xipengw@umich.edu

import sys
import tkinter as tk
from tkinter import *
import numpy as np
import random
import time

GAMMA = 0.9
ALPHA = 0.1
EPSILON = 0.5

class MDP(object):
    """
The ice surface is described using a grid like the following

    SFFF
    FHFH
    FFFH
    HFFG

S : starting point, safe
F : frozen surface, safe
H : hole, fall to your doom
G : goal, where the frisbee is located

The episode ends when you reach the goal or fall in a hole.
You receive a reward of 1 if you reach the goal, and zero otherwise.
    """
    def __init__(self):
        self.P = {0: {0: [(0.1, 0, 0.0), (0.8, 0, 0.0), (0.1, 4, 0.0)], 1: [(0.1, 0, 0.0), (0.8, 4, 0.0), (0.1, 1, 0.0)], 2: [(0.1, 4, 0.0), (0.8, 1, 0.0), (0.1, 0, 0.0)], 3: [(0.1, 1, 0.0), (0.8, 0, 0.0), (0.1, 0, 0.0)]}, 1: {0: [(0.1, 1, 0.0), (0.8, 0, 0.0), (0.1, 5, 0.0)], 1: [(0.1, 0, 0.0), (0.8, 5, 0.0), (0.1, 2, 0.0)], 2: [(0.1, 5, 0.0), (0.8, 2, 0.0), (0.1, 1, 0.0)], 3: [(0.1, 2, 0.0), (0.8, 1, 0.0), (0.1, 0, 0.0)]}, 2: {0: [(0.1, 2, 0.0), (0.8, 1, 0.0), (0.1, 6, 0.0)], 1: [(0.1, 1, 0.0), (0.8, 6, 0.0), (0.1, 3, 0.0)], 2: [(0.1, 6, 0.0), (0.8, 3, 0.0), (0.1, 2, 0.0)], 3: [(0.1, 3, 0.0), (0.8, 2, 0.0), (0.1, 1, 0.0)]}, 3: {0: [(0.1, 3, 0.0), (0.8, 2, 0.0), (0.1, 7, 0.0)], 1: [(0.1, 2, 0.0), (0.8, 7, 0.0), (0.1, 3, 0.0)], 2: [(0.1, 7, 0.0), (0.8, 3, 0.0), (0.1, 3, 0.0)], 3: [(0.1, 3, 0.0), (0.8, 3, 0.0), (0.1, 2, 0.0)]}, 4: {0: [(0.1, 0, 0.0), (0.8, 4, 0.0), (0.1, 8, 0.0)], 1: [(0.1, 4, 0.0), (0.8, 8, 0.0), (0.1, 5, 0.0)], 2: [(0.1, 8, 0.0), (0.8, 5, 0.0), (0.1, 0, 0.0)], 3: [(0.1, 5, 0.0), (0.8, 0, 0.0), (0.1, 4, 0.0)]}, 5: {0: [(1.0, 5, 0)], 1: [(1.0, 5, 0)], 2: [(1.0, 5, 0)], 3: [(1.0, 5, 0)]}, 6: {0: [(0.1, 2, 0.0), (0.8, 5, 0.0), (0.1, 10, 0.0)], 1: [(0.1, 5, 0.0), (0.8, 10, 0.0), (0.1, 7, 0.0)], 2: [(0.1, 10, 0.0), (0.8, 7, 0.0), (0.1, 2, 0.0)], 3: [(0.1, 7, 0.0), (0.8, 2, 0.0), (0.1, 5, 0.0)]}, 7: {0: [(1.0, 7, 0)], 1: [(1.0, 7, 0)], 2: [(1.0, 7, 0)], 3: [(1.0, 7, 0)]}, 8: {0: [(0.1, 4, 0.0), (0.8, 8, 0.0), (0.1, 12, 0.0)], 1: [(0.1, 8, 0.0), (0.8, 12, 0.0), (0.1, 9, 0.0)], 2: [(0.1, 12, 0.0), (0.8, 9, 0.0), (0.1, 4, 0.0)], 3: [(0.1, 9, 0.0), (0.8, 4, 0.0), (0.1, 8, 0.0)]}, 9: {0: [(0.1, 5, 0.0), (0.8, 8, 0.0), (0.1, 13, 0.0)], 1: [(0.1, 8, 0.0), (0.8, 13, 0.0), (0.1, 10, 0.0)], 2: [(0.1, 13, 0.0), (0.8, 10, 0.0), (0.1, 5, 0.0)], 3: [(0.1, 10, 0.0), (0.8, 5, 0.0), (0.1, 8, 0.0)]}, 10: {0: [(0.1, 6, 0.0), (0.8, 9, 0.0), (0.1, 14, 0.0)], 1: [(0.1, 9, 0.0), (0.8, 14, 0.0), (0.1, 11, 0.0)], 2: [(0.1, 14, 0.0), (0.8, 11, 0.0), (0.1, 6, 0.0)], 3: [(0.1, 11, 0.0), (0.8, 6, 0.0), (0.1, 9, 0.0)]}, 11: {0: [(1.0, 11, 0)], 1: [(1.0, 11, 0)], 2: [(1.0, 11, 0)], 3: [(1.0, 11, 0)]}, 12: {0: [(1.0, 12, 0)], 1: [(1.0, 12, 0)], 2: [(1.0, 12, 0)], 3: [(1.0, 12, 0)]}, 13: {0: [(0.1, 9, 0.0), (0.8, 12, 0.0), (0.1, 13, 0.0)], 1: [(0.1, 12, 0.0), (0.8, 13, 0.0), (0.1, 14, 0.0)], 2: [(0.1, 13, 0.0), (0.8, 14, 0.0), (0.1, 9, 0.0)], 3: [(0.1, 14, 0.0), (0.8, 9, 0.0), (0.1, 12, 0.0)]}, 14: {0: [(0.1, 10, 0.0), (0.8, 13, 0.0), (0.1, 14, 0.0)], 1: [(0.1, 13, 0.0), (0.8, 14, 0.0), (0.1, 15, 1.0)], 2: [(0.1, 14, 0.0), (0.8, 15, 1.0), (0.1, 10, 0.0)], 3: [(0.1, 15, 1.0), (0.8, 10, 0.0), (0.1, 13, 0.0)]}, 15: {0: [(1.0, 15, 0)], 1: [(1.0, 15, 0)], 2: [(1.0, 15, 0)], 3: [(1.0, 15, 0)]}}
        self.nS = 16 # number of states
        self.nA = 4 # number of actions
        self.world = {(0,0):0,(0,1):1,(0,2):1,(0,3):1,(1,0):1,(1,1):2,(1,2):1,(1,3):2,(2,0):1,(2,1):1,(2,2):1,(2,3):2,(3,0):2,(3,1):1,(3,2):1,(3,3):3}

def rl_tabular_q(mdp, alpha, gamma, epsilon, Qpi, s, userAction=None):
    #pick action
    if userAction == None:
        if(random.uniform(0,1) < epsilon):
            #pick random action
            action = random.randint(0, mdp.nA-1)
        else:
            action = np.argmax(Qpi[s])
    else:
        action = userAction

    p = random.uniform(0,1)
    #mdp.P[state][action] is a list of tuples (probability, nextstate, reward)
    rTupleList = mdp.P[s][action];
    pThresholds = []
    tmp = 0
    for i, rTuple in enumerate(rTupleList):
        tmp += rTuple[0]
        pThresholds.append(tmp)
    for i, rTuple in enumerate(rTupleList):
        if p <= pThresholds[i]:
            new_s = rTuple[1]
            reward = rTuple[2]
            break
    target = reward + gamma*np.max(Qpi[new_s])
    newQ = (1-alpha)*Qpi[s][action] + alpha*target
    Qpi[s][action] = newQ
    return (new_s,action)

colors = ['red', 'white', 'black', 'green']
def getBlack():
    return "black"

def getColor(r, g, b):
    return "#%0.2X%0.2X%0.2X" % (int(r*255), int(g*255), int(b*255))

class Cell():
    def __init__(self, master, x, y, size):
        """ Constructor of the object called by Cell(...) """
        self.master = master
        self.abs = x
        self.ord = y
        self.size= size

    def draw(self):
        """ order to the cell to draw its representation on the canvas """
        if self.master != None :
            vi = self.master.vi
            outline = getBlack()

            xmin = self.abs * self.size
            xmax = xmin + self.size
            ymin = self.ord * self.size
            ymax = ymin + self.size
            colorIdx = self.master.mdp.world[(self.ord,self.abs)]
            self.master.create_rectangle(xmin, ymin, xmax, ymax, fill = colors[colorIdx], outline = outline)
            _x = int(self.master.state % 4)
            _y = int(self.master.state / 4)
            if(self.ord == _y and self.abs == _x):
                #draw a yellow circle
                delta = self.size / 4.0
                self.master.create_oval(xmin + delta , ymin + delta, xmax - delta, ymax - delta, fill = "yellow", outline = 'blue')

            if True:
                p = vi[self.ord*4 + self.abs]
                action = np.argmax(p)
                if action == 0:
                    self.master.create_line(xmin + self.size/2, ymin + self.size/2, xmin+2*self.size/8, ymin+self.size/2, arrow=tk.LAST)
                elif action == 1:
                    self.master.create_line(xmin + self.size/2, ymin + self.size/2, xmin+self.size/2, ymax-2*self.size/8, arrow=tk.LAST)
                elif action == 2:
                    self.master.create_line(xmin + self.size/2, ymin + self.size/2, xmax-2*self.size/8, ymin+self.size/2, arrow=tk.LAST)
                elif action == 3:
                    self.master.create_line(xmin + self.size/2, ymin + self.size/2, xmin+self.size/2, ymin+2*self.size/8, arrow=tk.LAST)
                for i,v in enumerate(p):
                    if i == 0:
                        self.master.create_text(xmin+self.size/8, ymin+self.size/2, fill="black", font="Times "+str(int(self.master.cellSize/8))+" italic bold",text="%0.2f"%v)
                    if i == 1:
                        self.master.create_text(xmin+self.size/2, ymax-self.size/8, fill="black", font="Times "+str(int(self.master.cellSize/8))+" italic bold",text="%0.2f"%v)
                    if i == 2:
                        self.master.create_text(xmax-self.size/8, ymin+self.size/2, fill="black", font="Times "+str(int(self.master.cellSize/8))+" italic bold",text="%0.2f"%v)
                    if i == 3:
                        self.master.create_text(xmin+self.size/2, ymin+self.size/8, fill="black", font="Times "+str(int(self.master.cellSize/8))+" italic bold",text="%0.2f"%v)
                #print()

class CellGrid(Canvas):
    def __init__(self, master, mdp, vi, rowNumber, columnNumber, cellSize, *args, **kwargs):
        Canvas.__init__(self, master, width = cellSize * columnNumber , height = cellSize * rowNumber, *args, **kwargs)

        self.cellSize = cellSize
        self.mdp = mdp
        self.vi = vi
        self.grid = []
        self.state = 0
        self.iters = 0
        for row in range(rowNumber):
            line = []
            for column in range(columnNumber):
                line.append(Cell(self, column, row, cellSize))

            self.grid.append(line)

        #bind click action
        master.bind("<Key>", self.key)
        self.draw()

    def draw(self):
        for row in self.grid:
            for cell in row:
                cell.draw()


    def key(self, event):
        if(event.keycode == 8124162):
            #print("West")
            actions = ["West", "South", "East","North"]
            (self.state,action) = rl_tabular_q(mdp, ALPHA, GAMMA, EPSILON, self.vi, self.state, 0)
            print("Action: %s"%actions[action])
            #check if a terminate state
            x = int(self.state % 4)
            y = int(self.state / 4)
            if(self.mdp.world[(y,x)] == 2 or self.mdp.world[(y,x)] == 3):
                self.state = 0
            self.draw()
        if(event.keycode == 8189699):
            #print("East")
            actions = ["West", "South", "East","North"]
            (self.state,action) = rl_tabular_q(mdp, ALPHA, GAMMA, EPSILON, self.vi, self.state,2)
            print("Action: %s"%actions[action])
            #check if a terminate state
            x = int(self.state % 4)
            y = int(self.state / 4)
            if(self.mdp.world[(y,x)] == 2 or self.mdp.world[(y,x)] == 3):
                self.state = 0
            self.draw()
        if(event.keycode == 8320768):
            #print("North")
            actions = ["West", "South", "East","North"]
            (self.state,action) = rl_tabular_q(mdp, ALPHA, GAMMA, EPSILON, self.vi, self.state,3)
            print("Action: %s"%actions[action])
            #check if a terminate state
            x = int(self.state % 4)
            y = int(self.state / 4)
            if(self.mdp.world[(y,x)] == 2 or self.mdp.world[(y,x)] == 3):
                self.state = 0
            self.draw()
        if(event.keycode == 8255233):
            #print("South")
            actions = ["West", "South", "East","North"]
            (self.state,action) = rl_tabular_q(mdp, ALPHA, GAMMA, EPSILON, self.vi, self.state,1)
            print("Action: %s"%actions[action])
            #check if a terminate state
            x = int(self.state % 4)
            y = int(self.state / 4)
            if(self.mdp.world[(y,x)] == 2 or self.mdp.world[(y,x)] == 3):
                self.state = 0
            self.draw()
        if (event.char == ' '):
            actions = ["West", "South", "East","North"]
            (self.state,action) = rl_tabular_q(mdp, ALPHA, GAMMA, EPSILON, self.vi, self.state)
            print("Action: %s"%actions[action])
            #check if a terminate state
            x = int(self.state % 4)
            y = int(self.state / 4)
            if(self.mdp.world[(y,x)] == 2 or self.mdp.world[(y,x)] == 3):
                self.state = 0
            self.draw()

        if (event.char == 'n'):
            steps = 10000
            self.iters += steps
            print("Total %d iters: %f", self.iters, np.max([0.1, 1.0*EPSILON*steps/self.iters]))
            for i in range(0, steps):
                (self.state,action) = rl_tabular_q(mdp, ALPHA, GAMMA, np.max([0.1, 1.0*EPSILON*steps/self.iters]), self.vi, self.state)
                x = int(self.state % 4)
                y = int(self.state / 4)
                if(self.mdp.world[(y,x)] == 2 or self.mdp.world[(y,x)] == 3):
                    self.state = 0
            self.draw()

        if (event.char == 'q'):
            self.master.destroy()

if __name__ == "__main__" :

    mdp = MDP()
    print(mdp.__doc__)
    print("mdp.P is a two-level dict where the first key is the state and the second key is the action.")
    print("The 2D grid cells are associated with indices [0, 1, 2, ..., 15] from left to right and top to down, as in")
    print(np.arange(16).reshape(4,4))
    print("Action indices [0, 1, 2, 3] correspond to West, South, East and North.")
    print("mdp.P[state][action] is a list of tuples (probability, nextstate, reward).\n")
    print("For example, state 0 is the initial state, and the transition information for s=0, a=0 is \nP[0][0] =", mdp.P[0][0], "\n")
    print("As another example, state 5 corresponds to a hole in the ice, in which all actions lead to the same state with probability 1 and reward 0.")
    for i in range(4):
        print("P[5][%i] =" % i, mdp.P[5][i])

    # Value iteration
    actions = ["W", "S", "E", "N"]
    app = Tk()
    Qpi = np.zeros([mdp.nS, mdp.nA])
    grid = CellGrid(app, mdp, Qpi, 4, 4, 200)
    grid.pack()
    app.mainloop()
