#!/usr/bin/env python3
import enchant
import sys

def checkCondition(solution, position):
    return True

def solveCPS(values, size, checkCondition):
    """Finds a solution to a backtracking problem.

    values     -- a sequence of values to try, in order.
    size       -- the total number of “slots” you are trying to fill

    Return the solution as a list of values.
    """
    solutions = []
    solution = [None] * size

    def extend_solution(position):
        for value in values:
            solution[position] = value
            if not checkCondition(solution, position):
                continue
            if position >= size-1 or extend_solution(position+1):
                solutions.append(solution.copy())

    extend_solution(0)
    return solutions

solutions = []
values = sys.argv[2].split(" ");
solutions = solveCPS(values, int(sys.argv[1]), checkCondition)
d = enchant.Dict("en_US")
words = set()
for s in solutions:
    word = "".join(s)
    if(d.check(word)):
        words.add(word)
print(words)
