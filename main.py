from problem import Problem
from solver import PulpSolver
import numpy as np

input_filename = 'problem.csv'

if __name__ == '__main__':
    problem = Problem()
    problem.randomize(9)
    problem.saveCSV(input_filename)
    
    problem.readCSV(input_filename)
    solver = PulpSolver()
    schedule, reward = solver.solve(problem)
    
    print("Schedule (solution):")
    print(schedule)
    print(f"Reward {reward} out of {np.sum(problem.p)}")