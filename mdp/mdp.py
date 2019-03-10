#!/usr/bin/env python3
import numpy as np

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

def printOptActions(optimalActions):
    print("==================")
    print("Gridworld:")
    print("SFFF\nFHFH\nFFFH\nHFFG")
    print("Optimal actions:")
    for i in range(4):
        for j in range(4):
            print(optimalActions[i*4+j], end="")
        print()
    print("==================")

def main():
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

    GAMMA = 0.95
    # Value iteration
    actions = ["W", "S", "E", "N"]
    Vs_VI, pis_VI = value_iteration(mdp, GAMMA, 20)
    #QVs_VI, pis_VI =qvalue_iteration(mdp,GAMMA, 20)
    #Vs_PI, pis_PI = policy_iteration(mdp, GAMMA, 20)
    optimalActions = [actions[e] for e in pis_VI[-1]]
    printOptActions(optimalActions)


if __name__ == "__main__" :
    main()
