#!/usr/bin/env python3

from tkinter import *
import numpy as np
import random


HEIGHT = 6
WIDTH = 6
CELLSIZE = 80
pInit = 1.0 / HEIGHT / WIDTH
prob = np.full((HEIGHT,WIDTH), pInit)
ghost = [random.randint(0, WIDTH-1), random.randint(0,HEIGHT-1)]
last_selection = [0,0]
colors = ["red","green","yellow","blue"]

#Random
pDistantZero = [0.8, 0.15, 0.04, 0.01]
pDistantOne = [0.1, 0.6, 0.2, 0.1]
pDistantTwoThree = [0.02,0.08,0.8,0.1]
pDistantAboveThree = [0.02,0.03,0.1,0.85]

def getObserveColor(p, pDistant):
    lowBoundary = 0
    for i in range(0, len(pDistant)):
        if(p <= pDistant[i]+lowBoundary and p >= lowBoundary):
            return colors[i]
        lowBoundary = pDistant[i]+lowBoundary


def probe(x,y):
    dist = abs(x-ghost[0]) + abs(y-ghost[1])
    p = random.uniform(0, 1)
    if(dist == 0):
        return getObserveColor(p, pDistantZero)
    elif(dist == 1):
        return getObserveColor(p, pDistantOne)
    elif(dist == 2 or dist == 3):
        return getObserveColor(p, pDistantTwoThree)
    else:
        return getObserveColor(p, pDistantAboveThree)

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
        self.selected = False

    def setObserveColor(self, color):
        self.selected = True
        self.selectedColor = color

    def draw(self):
        """ order to the cell to draw its representation on the canvas """
        if self.master != None :
            p = prob[self.ord][self.abs]
            fill = getColor(p, 1-p, 0)
            outline = getBlack()

            xmin = self.abs * self.size
            xmax = xmin + self.size
            ymin = self.ord * self.size
            ymax = ymin + self.size
            self.master.create_rectangle(xmin, ymin, xmax, ymax, fill = "white", outline = outline)
            self.master.create_rectangle(xmin+self.size/4.0, ymin+self.size/4, xmax-self.size/4.0, ymax-self.size/4, fill = fill, outline = outline)
            self.master.create_text(xmin+self.size/2, ymin+self.size/2, fill="black", font="Times "+str(int(self.master.cellSize/4))+" italic bold",text="%0.2f"%p)
        if self.selected:
            self.master.create_oval(xmin, ymin, xmax, ymax, fill = None, outline = self.selectedColor, width = 3)

class CellGrid(Canvas):
    def __init__(self, master, rowNumber, columnNumber, cellSize, *args, **kwargs):
        Canvas.__init__(self, master, width = cellSize * columnNumber , height = cellSize * rowNumber, *args, **kwargs)
        self.score = rowNumber * columnNumber
        self.cellSize = cellSize

        self.grid = []
        for row in range(rowNumber):

            line = []
            for column in range(columnNumber):
                line.append(Cell(self, column, row, cellSize))

            self.grid.append(line)

        #bind click action
        self.bind("<Button-1>", self.handleMouseClick)
        master.bind("<Key>", self.key)
        self.draw()



    def draw(self):
        for row in self.grid:
            for cell in row:
                cell.draw()

    def key(self, event):
        kp = repr(event.char)
        if (event.char == ' '):
            if ghost == last_selection:
                print("Final score: ", end='')
                print(self.score)
                self.master.destroy()
            else:
                print("Wrong")

    def _eventCoords(self, event):
        row = int(event.y / self.cellSize)
        column = int(event.x / self.cellSize)
        return row, column

    # TODO: Complete this prob update function
    def updateProb(self, color, x, y):
        bel = prob
        #for w in range(0, WIDTH):
            #for h in range(0, HEIGHT):
                #TODO: Update probability of element in row h and column w.
                #bel[h][w] = ?
        for h in range(0, HEIGHT):
            for w in range(0, WIDTH):
                print("%0.2f" % bel[h][w], end=',')
            print()
        print('==================')
    def handleMouseClick(self, event):
        row, column = self._eventCoords(event)
        cell = self.grid[row][column]
        color = probe(column, row)
        cell.setObserveColor(color)
        last_selection[:] = [column, row]
        self.updateProb(color, column, row)
        self.draw()
        #cell.draw()
        self.score = self.score - 1


if __name__ == "__main__" :
    app = Tk()
    grid = CellGrid(app, HEIGHT, WIDTH, CELLSIZE)
    grid.pack()

    app.mainloop()
