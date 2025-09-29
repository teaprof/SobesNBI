# Compares the extended pulp solver solutions with solutions from bruteforce solver
from main_ext import ExtendedProblem, ExtendedPulpSolver
from crash_test import BruteforceSolver
import numpy as np       
            

if __name__ == '__main__':
    # number of samples
    Nsamples = 10000
    
    # fix seed for debug purposes
    np.random.seed(1)
    seeds = np.random.randint(0, 1000000000, Nsamples)
    
    # create both solvers and start crash test
    bruteforceSolver = BruteforceSolver()
    pulpSolver = ExtendedPulpSolver()
    for n in range(Nsamples):
        print(f"seeds[{n}] = {seeds[n]}")
        np.random.seed(seeds[n])
        
        # create random problem
        p = ExtendedProblem()
        p.randomize()
        print(p)
        
        # call brute-force solver
        schedule_0, reward_0 = bruteforceSolver.solve(p)
        print(schedule_0)
        print(f"objective: {reward_0}")

        # call pulp solver
        schedule, reward = pulpSolver.solve(p)
        pulpSolver.debugPrint(p)
        
        # compare solutions
        assert p.checkSchedule(schedule)
        print(f"objective: {p.objectiveFcn(schedule)}")
        assert np.isclose(reward, reward_0, 1e-10, 1e-10)
    print("\n\n")

    # print statistics
    print("All tests are Ok")        
    print(f"Bruteforce solver time: {bruteforceSolver.elapsed}")
    print(f"PuLP solver time: {pulpSolver.elapsed}")
