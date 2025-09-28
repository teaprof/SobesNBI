# Compare the pulp solver solutions with solutions from bruteforce solver
from problem import Problem
from solver import PulpSolver
import numpy as np
import itertools
import time 
        

class BruteforceSolver:
    def __init__(self):
        self.elapsed = 0
        
    def solve(self, p: Problem): #-> tuple(list[int], float):
        self.elapsed -= time.time()
        # initialize starting solution
        schedule0 = list(range(p.nProducts))
        best_schedule = schedule0
        best_reward = p.objectiveFcn(schedule0)
        
        # calculate the max possible value of the total reward, this is
        # used to stop the bruteforce loop if this values is reached
        max_possible_value = np.sum([p.p[i] for i in range(p.nProducts) if p.t[i] < p.d[i]])            
        
        # main loop - enumerate all possible schedules        
        for schedule in itertools.permutations(schedule0):
            if not p.checkSolution(schedule):
                continue
            sched_as_list = list(schedule)
            cur_reward = p.objectiveFcn(sched_as_list)
            if cur_reward > best_reward:
                # update maximum
                best_reward = cur_reward
                best_schedule = sched_as_list
                if best_reward == max_possible_value:
                    break
        self.elapsed += time.time()
        return best_schedule, best_reward
            

if __name__ == '__main__':
    # number of samples
    Nsamples = 10000
    
    # fix seed for debug purposes
    np.random.seed(1)
    seeds = np.random.randint(0, 1000000000, Nsamples)
    
    # create both solvers and start crash test
    bruteforceSolver = BruteforceSolver()
    pulpSolver = PulpSolver()
    for n in range(Nsamples):
        print(f"seeds[{n}] = {seeds[n]}")
        np.random.seed(seeds[n])
        
        # create random problem
        p = Problem()
        p.randomize()
        print(p)
        
        # call bruteforce
        schedule_0, reward_0 = bruteforceSolver.solve(p)
        print(schedule_0)
        print(f"objective: {reward_0}")

        # call pulp solver
        schedule, reward = pulpSolver.solve(p)
        pulpSolver.debugPrint(p)
        
        # compare solutions
        assert p.checkSolution(schedule)
        print(f"objective: {p.objectiveFcn(schedule)}")
        assert np.isclose(reward, reward_0, 1e-10, 1e-10)
    print("\n\n")

    # print statistics
    print("All tests are Ok")        
    print(f"Bruteforce solver time: {bruteforceSolver.elapsed}")
    print(f"PuLP solver time: {pulpSolver.elapsed}")
