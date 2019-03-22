#!/usr/bin/env python3

#EECS 492(Winter 2019) Discussion
#xipengw@umich.edu

import sys
import tkinter as tk
from tkinter import *
import numpy as np
import random

class MDP(object):
    def __init__(self):
        self.P = {0: {0: [(1, 0, 0.0)], 1: [(1, 0, 0.0)]},
                  1: {0: [(0.8, 0, -1.0), (0.2, 2, 0.0)], 1: [(0.8, 2, 0.0), (0.2, 0, -1.0)]},
                  2: {0: [(0.8, 1, 0.0), (0.2, 3, 1.0)], 1: [(0.8, 3, 1.0), (0.2, 1, 0.0)]},
                  3: {0: [(1, 3, 0.0)], 1: [(1, 3, 0.0)]}}
        self.nS = 4 # number of states
        self.nA = 2 # number of actions
        self.world = {(0,0):2,(0,1):1,(0,2):1,(0,3):3}

def value_iteration(mdp, gamma, nIt):
    """
    Inputs:
        mdp: MDP
        gamma: discount factor
        nIt: number of iterations, corresponding to n above
    Outputs:
        (value_functions, policies)

    len(value_functions) == nIt+1 and len(policies) == nIt
    """
    Vs = [np.zeros(mdp.nS)] # list of value functions contains the initial value function V^{(0)}, which is zero
    pis = []
    for it in range(nIt):
        oldpi = pis[-1] if len(pis) > 0 else None
        Vprev = Vs[-1]
        V = np.zeros(mdp.nS)
        pi = np.zeros(mdp.nS, dtype=int)
        for i,v in enumerate(Vprev):
            Qvalue = []
            for key in mdp.P[i]:
                v = 0
                for e in mdp.P[i][key]:
                    v += e[0] * (e[2] + gamma * Vprev[e[1]])
                Qvalue.append(v)
            pi[i] = int(np.argmax(Qvalue))
            V[i] = Qvalue[pi[i]]
        Vs.append(V)
        pis.append(pi)
    return Vs, pis

def qvalue_iteration(mdp,gamma, nIt):
    QVs = []
    pis = []

    Qpi = np.zeros([mdp.nS, mdp.nA])
    QVs.append(Qpi)
    for it in range(nIt):
        oldpi = pis[-1] if len(pis) > 0 else None
        Vprev = QVs[-1] #
        V = np.zeros([mdp.nS, mdp.nA])
        pi = np.zeros(mdp.nS, dtype=int)
        for i in range(mdp.nS):
            for j in range(mdp.nA):
                v = 0
                for e in mdp.P[i][j]:
                    v += e[0] * (e[2] + gamma * np.max(Vprev[e[1]]))
                V[i][j] = v
        QVs.append(V)
        for i in range(mdp.nS):
            pi[i] = np.argmax(V[i])
        pis.append(pi)

    return QVs, pis

def compute_vpi(pi, mdp, gamma):
    # use pi[state] to access the action that's prescribed by this policy
    a = np.full((mdp.nS, mdp.nS), 0.0)
    b = np.full(mdp.nS, 0.0)
    for row in range(mdp.nS):
        a[row][row] += 1
        for e in mdp.P[row][pi[row]]:
            # (p, s', r)
            b[row] += e[0] * e[2]
            a[row][int(e[1])] = a[row][int(e[1])] + (-gamma*e[0])
    V = np.linalg.solve(a, b)
    return V

def compute_qpi(vpi, mdp, gamma):
    Qpi = np.zeros([mdp.nS, mdp.nA])
    for i in range(mdp.nS):
        for key in mdp.P[i]:
            v = 0
            for e in mdp.P[i][key]:
                v += e[0] * (e[2] + gamma * vpi[e[1]])
            Qpi[i][key] = v
    return Qpi

def policy_iteration(mdp, gamma, nIt):
    Vs = []
    pis = []
    pi_prev = np.zeros(mdp.nS,dtype='int')
    pis.append(pi_prev)
    for it in range(nIt):
        # you need to compute qpi which is the state-action values for current pi
        vpi = compute_vpi(pi_prev, mdp, gamma)
        qpi = compute_qpi(vpi, mdp, gamma)
        pi = qpi.argmax(axis=1)
        Vs.append(vpi)
        pis.append(pi)
        pi_prev = pi
    return Vs, pis

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
            pi =  self.master.pi
            p = vi[self.ord*4 + self.abs]
            if(type(p) == np.float64):
                fill = getColor(p, 1-p, 0)
            else:
                fill = getColor(np.max(p), 1-np.max(p), 0)
            fill = getColor(0, 1, 0)
            outline = getBlack()

            xmin = self.abs * self.size
            xmax = xmin + self.size
            ymin = self.ord * self.size
            ymax = ymin + self.size
            colorIdx = self.master.mdp.world[(self.ord,self.abs)]
            self.master.create_rectangle(xmin, ymin, xmax, ymax, fill = colors[colorIdx], outline = outline)
            if(type(p) == np.float64):
                if type(pi) != type(-1):
                    action = pi[self.ord][self.abs]
                    if action == 0:
                        self.master.create_line(xmin + self.size/2, ymin + self.size/2, xmin, ymin+self.size/2, arrow=tk.LAST)
                    elif action == 1:
                        self.master.create_line(xmin + self.size/2, ymin + self.size/2, xmax, ymin+self.size/2, arrow=tk.LAST)
                self.master.create_rectangle(xmin+self.size/4.0, ymin+self.size/4, xmax-self.size/4.0, ymax-self.size/4, fill = fill, outline = outline)
                self.master.create_text(xmin+self.size/2, ymin+self.size/2, fill="black", font="Times "+str(int(self.master.cellSize/8))+" italic bold",text="%0.4f"%p)
            else:
                action = np.argmax(p)
                if action == 0:
                    self.master.create_line(xmin + self.size/2, ymin + self.size/2, xmin+2*self.size/8, ymin+self.size/2, arrow=tk.LAST)
                elif action == 1:
                    self.master.create_line(xmin + self.size/2, ymin + self.size/2, xmax-2*self.size/8, ymin+self.size/2, arrow=tk.LAST)
                for i,v in enumerate(p):
                    if i == 0:
                        self.master.create_text(xmin+self.size/8, ymin+self.size/2, fill="black", font="Times "+str(int(self.master.cellSize/8))+" italic bold",text="%0.4f"%v)
                    if i == 1:
                        self.master.create_text(xmax-self.size/8, ymin+self.size/2, fill="black", font="Times "+str(int(self.master.cellSize/8))+" italic bold",text="%0.4f"%v)
                #print()
class CellGrid(Canvas):
    def __init__(self, master, mdp, maxIter, vis, pis, rowNumber, columnNumber, cellSize, *args, **kwargs):
        Canvas.__init__(self, master, width = cellSize * columnNumber , height = cellSize * rowNumber, *args, **kwargs)
        self.score = rowNumber * columnNumber
        self.cellSize = cellSize
        self.mdp = mdp
        self.vis = vis
        self.pis = pis
        self.maxIter = maxIter
        self.iter = 0
        self.vi = self.vis[self.iter]
        self.pi = -1
        self.grid = []
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
        kp = repr(event.char)
        if (event.char == ' '):
            if self.iter < self.maxIter:
                self.iter += 1
                self.vi = self.vis[self.iter]
                self.pi = self.pis[self.iter].reshape(1,4)
                self.draw()
            else:
                print("Please increase maxIter number in the code")
        if (event.char == 'q'):
            self.master.destroy()

if __name__ == "__main__" :
    if len(sys.argv) < 2:
        print("./main.py [value_iter, policy_iter, q_iter]")
        exit(0)

    mdp = MDP()
    GAMMA = 0.9
    ITER = 20
    # Value iteration
    actions = ["W", "S", "E", "N"]
    app = Tk()
    if(sys.argv[1] == 'value_iter'):
        Vs_VI, pis_VI = value_iteration(mdp, GAMMA, ITER)
        grid = CellGrid(app, mdp, ITER, Vs_VI, pis_VI, 1, 4, 200)
    elif(sys.argv[1] == 'policy_iter'):
        Vs_PI, pis_PI = policy_iteration(mdp, GAMMA, ITER)
        grid = CellGrid(app, mdp, ITER, Vs_PI, pis_PI, 1, 4, 200)
    elif(sys.argv[1] == 'q_iter'):
        QVs_VI, pis_VI = qvalue_iteration(mdp,GAMMA, ITER)
        grid = CellGrid(app, mdp, ITER, QVs_VI, pis_VI, 1, 4, 200)
    else:
        print("./main.py [value_iter, policy_iter, q_iter]")
        exit(0)
    grid.pack()
    app.mainloop()
