#!/usr/bin/env python3
import numpy as np
from enum import Enum
from random import randint
from random import uniform

class AgentType(Enum):
    PACMAN = 1
    GHOST = 2

def checkActionValid(x,y,w,h,walls):
    if(0 <= x and x < w and 0 <= y and y < h and (not walls[y][x])):
        return True
    return False

def ghostActions(agent, gridWorld):
    print("Ghost moved:", end=' ')
    # Random walk
    while(True):
        x = agent.pos[0]
        y = agent.pos[1]
        w = gridWorld.w
        h = gridWorld.h
        walls = gridWorld.walls
        dir = agent.dir
        p = uniform(0, 1)
        # We have 30% chance go random direction instead of following
        if(p > 0.7):
            agent.hitWall = True
        if agent.hitWall:
            dir = randint(0, 3)
        if(dir == 0):
            if(checkActionValid(x - 1, y, w, h, walls)):
                print("go left")
                agent.pos[0] = x - 1
                agent.pos[1] = y
                agent.dir = dir
                agent.hitWall = False
                return
            else:
                agent.hitWall = True
        elif(dir == 1):
            if(checkActionValid(x + 1, y, w, h, walls)):
                print("go right")
                agent.pos[0] = x + 1
                agent.pos[1] = y
                agent.dir = dir
                agent.hitWall = False
                return
            else:
                agent.hitWall = True
        elif(dir == 2):
            if(checkActionValid(x, y-1, w, h, walls)):
                print("go up")
                agent.pos[0] = x
                agent.pos[1] = y-1
                agent.dir = dir
                agent.hitWall = False
                return
            else:
                agent.hitWall = True
        elif(dir == 3):
            if(checkActionValid(x, y+1, w, h, walls)):
                print("go down")
                agent.pos[0] = x
                agent.pos[1] = y+1
                agent.dir = dir
                agent.hitWall = False
                return
            else:
                agent.hitWall = True

def randomActions(agent, gridWorld):
    # Random walk
    if agent.type == AgentType.PACMAN:
        print("Pacman moved")
    else:
        print("Ghost moved")
    while(True):
        dir = randint(0, 3)
        x = agent.pos[0]
        y = agent.pos[1]
        w = gridWorld.w
        h = gridWorld.h
        walls = gridWorld.walls
        if(dir == 0):
            # go left
            if(checkActionValid(x - 1, y, w, h, walls)):
                agent.pos[0] = x - 1
                agent.pos[1] = y
                return
        elif(dir == 1):
            # go right
            if(checkActionValid(x + 1, y, w, h, walls)):
                agent.pos[0] = x + 1
                agent.pos[1] = y
                return
        elif(dir == 2):
            # go up
            if(checkActionValid(x, y-1, w, h, walls)):
                agent.pos[0] = x
                agent.pos[1] = y-1
                return
        elif(dir == 3):
            # go down
            if(checkActionValid(x, y+1, w, h, walls)):
                agent.pos[0] = x
                agent.pos[1] = y+1
                return

class Node:
    def __init__(self, x, y, cost, par):
        self.pos = [x, y]
        self.par = par
        self.cost = cost
# Dijkstra
def findShortestPath(s, g, w, h, walls):
    startNode = Node(s[0], s[1], 0, None)
    priorityQ = [startNode]
    visited = np.full((w,h),0)
    visitedList = np.full((w,h),0)
    visited[startNode.pos[0]][startNode.pos[1]]
    path = []
    while(len(priorityQ)):
        currNode = min(priorityQ, key = lambda t: t.cost)
        priorityQ.remove(currNode)
        visitedList[currNode.pos[0]][currNode.pos[1]] = 1
        if(currNode.pos == g):
            node = currNode
            #print("Found")
            path.insert(0, node.pos);
            while(node.par):
                node = node.par;
                path.insert(0, node.pos);
            return path
        x = currNode.pos[0]
        y = currNode.pos[1]
        cost = currNode.cost + 1
        newX = x - 1
        newY = y
        if(checkActionValid(newX, newY, w, h, walls) and not visitedList[newX][newY]):
            # add new node
            if(visited[newX][newY]):
                #check q whether exist
                for e in priorityQ:
                    if(e.pos[0] == newX and e.pos[1] == newY and e.cost > cost):
                        e.cost = cost
                        e.par = currNode
            else:
                priorityQ.append(Node(newX,newY,cost,currNode))
                visited[newX][newY] = 1
        newX = x + 1
        newY = y
        if(checkActionValid(newX, newY, w, h, walls) and not visitedList[newX][newY]):
            # add new node
            if(visited[newX][newY]):
                #check q whether exist
                for e in priorityQ:
                    if(e.pos[0] == newX and e.pos[1] == newY and e.cost > cost):
                        e.cost = cost
                        e.par = currNode
            else:
                priorityQ.append(Node(newX,newY,cost,currNode))
                visited[newX][newY] = 1
        newX = x
        newY = y-1
        if(checkActionValid(newX, newY, w, h, walls) and not visitedList[newX][newY]):
            # add new node
            if(visited[newX][newY]):
                #check q whether exist
                for e in priorityQ:
                    if(e.pos[0] == newX and e.pos[1] == newY and e.cost > cost):
                        e.cost = cost
                        e.par = currNode
            else:
                priorityQ.append(Node(newX,newY,cost,currNode))
                visited[newX][newY] = 1
        newX = x
        newY = y+1
        if(checkActionValid(newX, newY, w, h, walls) and not visitedList[newX][newY]):
            # add new node
            if(visited[newX][newY]):
                #check q whether exist
                for e in priorityQ:
                    if(e.pos[0] == newX and e.pos[1] == newY and e.cost > cost):
                        e.cost = cost
                        e.par = currNode
            else:
                priorityQ.append(Node(newX,newY,cost,currNode))
                visited[newX][newY] = 1
    return None

def pacManActions(agent, gridWorld):
    print("PacMan moved")
    # Find closest foods
    minDist = 100000000000
    food = [-1,-1]
    for x in range(gridWorld.w):
        for y in range(gridWorld.h):
            if(gridWorld.foods[y][x]):
                dist = abs(agent.pos[0] - x) + abs(agent.pos[1] - y)
                if(dist < minDist):
                    minDist = dist
                    food[0] = x
                    food[1] = y
    if(food[0] == -1 and food[1] == -1):
        # No more food
        return

    # find the shortest path to food
    path = findShortestPath(agent.pos, food, gridWorld.w, gridWorld.h, gridWorld.walls)
    #print(path)
    if(path):
        agent.pos = path[1]

#TODO:
def pacManIntelligentActions(agent, gridWorld):
    # agent.pos = ???
    return

class Agent():
    def __init__(self, ID, color, initialX, initialY, agentType, actions):
        self.ID = ID
        self.color = color
        self.pos = [initialX, initialY]
        self.type = agentType
        self.action = actions
        self.hitWall = True
        self.dir = 0
    def move(self, gridWorld):
        self.action(self, gridWorld)

class GridWorld():
    """
    A tile-based world with a single agent.
    """
    def __init__(self, h, w, walls):
        self.h = h
        self.w = w
        self.score = 0
        self.walls = walls
        self.foods = np.full((h,w),0)
        self.agents = []
        # Add pacman agent
        self.add_agent(Agent(0, "yellow", 0, 0, AgentType.PACMAN, pacManActions))
        #self.add_agent(Agent(0, 0, AgentType.PACMAN, randomActions))
        for i in range(self.h):
            for j in range(self.w):
                if self.walls[i][j]:
                    self.foods[i][j] = 0
                else:
                    self.foods[i][j] = 1
        self.foods[0][0] = 0

    def add_agent(self,agent):
        self.agents.append(agent)

    def simulate(self):
        if(sum(sum(self.foods)) == 0):
            # Clear all food
            return True
        #update foods
        lastPositions = []
        for agent in self.agents:
            lastPositions.append(agent.pos.copy())
            agent.move(self)
            if(agent.type == AgentType.PACMAN):
                if(self.foods[agent.pos[1]][agent.pos[0]]):
                    self.foods[agent.pos[1]][agent.pos[0]] = 0
                    self.score = self.score + 5
        self.score = self.score - 1
        print("Score: %d\n" % self.score)
        pacMan = self.agents[0]
        for i,agent in enumerate(self.agents):
            if(agent.type == AgentType.GHOST):
                if agent.pos == pacMan.pos:
                    return True
                if agent.pos == lastPositions[0] and lastPositions[i] == pacMan.pos:
                    return True
        return False
