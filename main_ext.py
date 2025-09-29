from grammar_parser import GrammarParser
from problem_ext import ExtendedProblem
from solver_ext import ExtendedPulpSolver
from crash_test import BruteforceSolver
import numpy as np

if __name__ == '__main__':
    # Read file
    filename = "problem.schedlang"
    with open(filename, "r") as f:
        description = f.read()
        
    # Create ExtendedProblem
    parser = GrammarParser()
    ext_problem: ExtendedProblem = parser.parse(description)
    
    # Create and run solver
    solver = ExtendedPulpSolver()    
    schedule, reward = solver.solve(ext_problem)
    
    # Check solution
    assert ext_problem.checkSchedule(schedule)
    print("Schedule (solution):")
    print(schedule)
    print(f"Reward {reward} out of {np.sum(ext_problem.p)}")

    # Compare with BruteforceSolver solution
    bruteforceSolver = BruteforceSolver()
    _, reward0 = bruteforceSolver.solve(ext_problem)

    print(f"Bruteforce solution reward: {reward0}")

    assert np.isclose(reward, reward0, 1e-5)
    print("All Ok")
