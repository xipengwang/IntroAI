#!/usr/bin/env python3

#EECS 492(Winter 2019) Discussion
#xipengw@umich.edu

import sys
import numpy as np
from tkinter import *
from gridworld import *

class Cell():
    FILLED_COLOR_BG = "black"
    EMPTY_COLOR_BG = "white"
    FILLED_COLOR_BORDER = "black"
    EMPTY_COLOR_BORDER = "black"

    def __init__(self, master, x, y, size):
        """ Constructor of the object called by Cell(...) """
        self.master = master
        self.abs = x
        self.ord = y
        self.size= size

    def draw(self):
        """ order to the cell to draw its representation on the canvas """
        if self.master != None :
            fill = Cell.FILLED_COLOR_BG
            outline = Cell.FILLED_COLOR_BORDER

            xmin = self.abs * self.size
            xmax = xmin + self.size
            ymin = self.ord * self.size
            ymax = ymin + self.size

            if not self.master.gridWorld.walls[self.ord][self.abs]:
                fill = Cell.EMPTY_COLOR_BG
                outline = Cell.EMPTY_COLOR_BORDER

            self.master.create_rectangle(xmin, ymin, xmax, ymax, fill = fill, outline = outline)

            if self.master.gridWorld.foods[self.ord][self.abs]:
                delta = 7.0 * self.size / 16.0
                self.master.create_oval(xmin + delta , ymin + delta, xmax - delta, ymax - delta, fill = "blue", outline = 'blue')

            for agent in self.master.gridWorld.agents:
                if agent.type == AgentType.PACMAN:
                    if [self.abs, self.ord] == agent.pos:
                       delta = self.size / 4.0
                       self.master.create_oval(xmin + delta , ymin + delta, xmax - delta, ymax - delta, fill = agent.color, outline = 'blue')
                       self.master.create_text(xmin+self.size/2, ymin+self.size/2, fill="black", font="Times "+str(int(self.master.cellSize/4))+" italic bold",text=str(agent.ID))
                else:
                    if [self.abs, self.ord] == agent.pos:
                       delta = self.size / 4.0
                       self.master.create_oval(xmin + delta , ymin + delta, xmax - delta, ymax - delta, fill = agent.color, outline = 'green')
                       self.master.create_text(xmin+self.size/2, ymin+self.size/2, fill="black", font="Times "+str(int(self.master.cellSize/4))+" italic bold",text=str(agent.ID))

class CellGrid(Canvas):
    def __init__(self, master, gridWorld, rowNumber, columnNumber, cellSize, *args, **kwargs):
        Canvas.__init__(self, master, width = cellSize * columnNumber , height = cellSize * rowNumber, *args, **kwargs)

        self.cellSize = cellSize
        self.gridWorld = gridWorld

        self.grid = []
        for row in range(rowNumber):

            line = []
            for column in range(columnNumber):
                line.append(Cell(self, column, row, cellSize))

            self.grid.append(line)

        master.bind("<Key>", self.key)
        self.draw()


    def draw(self):
        for row in self.grid:
            for cell in row:
                cell.draw()


    def key(self, event):
        kp = repr(event.char)
        if (event.char == ' '):
            if(self.gridWorld.simulate() == True):
                print("Finished")
                self.draw()

                self.master.destroy()
            else:
                self.draw()
        if (event.char == 'q'):
            self.master.destroy()


def read_world_model(fileName):
    first = True
    i = 0
    with open(fileName) as fp:
        for line in fp.readlines():
            lineList = line.strip().split(",")
            if first:
                GRIDWORLD_H = int(lineList[0])
                GRIDWORLD_W = int(lineList[1])
                walls = np.full((GRIDWORLD_H, GRIDWORLD_W), 0)
                first = False
            else:
                for k in range(len(lineList)-1):
                    walls[i][k] = int(lineList[k])
                i = i+1
    return [walls, GRIDWORLD_H, GRIDWORLD_W]

if __name__ == "__main__" :
    cellSize = 80
    app = Tk()
    if(len(sys.argv) == 2):
        [walls, GRIDWORLD_H, GRIDWORLD_W] = read_world_model(sys.argv[1])
    else:
        [walls, GRIDWORLD_H, GRIDWORLD_W] = read_world_model('world.txt')
    gridWorld = GridWorld(GRIDWORLD_H, GRIDWORLD_W, walls)
    gridWorld.add_agent(Agent(1, 'green', 9, 9, AgentType.GHOST, ghostActions))
    gridWorld.add_agent(Agent(2, 'green', 0, 9, AgentType.GHOST, ghostActions))
    #gridWorld.add_agent(Agent(2, 'green', 0, 9, AgentType.GHOST, randomActions))
    grid = CellGrid(app, gridWorld, GRIDWORLD_H, GRIDWORLD_W, cellSize)
    grid.pack()
    app.mainloop()
