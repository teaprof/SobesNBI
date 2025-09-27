import pulp
import numpy as np
from numpy import typing as nt
import dataclasses
import itertools
import time 


@dataclasses.dataclass
class Problem:
    nMachines: int = 9
    t: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    d: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    p: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    
    def randomize(self, nMachines = None):
        if nMachines is None:
            self.nMachines = np.random.randint(1,10)
        else:
            self.nMachines = nMachines
        self.t = np.random.rand(self.nMachines)
        self.d = np.random.rand(self.nMachines)*self.nMachines/2
        #self.p = np.ones(self.nMachines) 
        self.p = np.random.rand(self.nMachines)            
    
    def __str__(self):
        str1 = f"nMachines = {self.nMachines}\n"
        str2 = "t = " + str(self.t)
        str3 = "d = " + str(self.d)
        str4 = "p = " + str(self.p)
        return str1 + "\n" + str2 + "\n" + str3 + "\n" + str4
    
    def targetfcn(self, schedule: list[int]) -> float:
        schedule = list(schedule)
        assert len(schedule) == self.nMachines
        assert len(set(schedule)) == self.nMachines
        tend = np.cumsum(self.t[schedule])
        bonus = 0
        for n in range(self.nMachines):
            if tend[n] < self.d[schedule[n]]:
                bonus += self.p[schedule[n]]
        return bonus
        # This version is slower approx 2 times:
        #bonus = np.sum(self.p[schedule]*(tend - self.d[schedule] <= 0))
        #return bonus                
    
    def checkSolution(self, schedule: list[int]) -> bool:
        if len(schedule) != self.nMachines:
            return False
        if len(set(schedule)) != self.nMachines:
            return False
        return True
        

class BruteforceSolver:
    def __init__(self):
        self.elapsed = 0

    def solve(self, p: Problem): #-> tuple(list[int], float):
        self.elapsed -= time.time()
        schedule0 = list(range(p.nMachines))
        best_schedule = schedule0
        best_value = p.targetfcn(schedule0)
        max_possible_value = np.sum([p.p[i] for i in range(p.nMachines) if p.t[i] < p.d[i]])    
        #max_possible_value = np.sum(p.p)
        for schedule in itertools.permutations(schedule0):
            lschedule = list(schedule)
            cur_value = p.targetfcn(lschedule)
            if cur_value > best_value:
                best_value = cur_value
                best_schedule = lschedule
                if best_value == max_possible_value:
                    break
        self.elapsed += time.time()
        return best_schedule, best_value
            
    
class PulpSolver:
    def __init__(self):
        self.elapsed = 0
        self.x = None
        self.problem = None
        self.u = None
        self.M = None
        self.M2 = None
        pass
    
    def solve(self, p: Problem) -> tuple[list[int], float]:
        self.problem = pulp.LpProblem("Scheduler problem", pulp.LpMaximize)
        #i - позиция в очереди, j - номер станка
        all_variables = [(i,j) for i in range(p.nMachines) for j in range(p.nMachines)]
        self.x = pulp.LpVariable.dicts("x", all_variables, cat=pulp.const.LpBinary)
        for i in range(p.nMachines):
                self.problem += pulp.lpSum([self.x[i,j] for j in range(p.nMachines)]) == 1
                self.problem += pulp.lpSum([self.x[j,i] for j in range(p.nMachines)]) == 1
                
            
        t_finish = 0
        self.M2 = 2*np.max(p.d)
        self.M = 2*(np.sum(p.t) + np.max(p.d))
        self.u = pulp.LpVariable.dicts("u", all_variables, cat = pulp.const.LpBinary)
        for i, j in all_variables:
                duration = self.x[i,j]*p.t[j]
                deadline = p.d[j] - self.M2*(1-self.x[i,j])
                t_finish = t_finish + duration
                self.problem += deadline - t_finish - self.M*self.u[i,j] <= 0
                self.problem += deadline - t_finish + self.M*(1-self.u[i,j]) >= 0

        self.problem += pulp.lpSum([self.u[i,j]*p.p[j] for i, j in all_variables])

        self.problem.solve()
        self.elapsed += self.problem.solutionTime
        self.problem.roundSolution()
        
        schedule: list[int] = []
        for i in range(p.nMachines):
            n = 0
            for j in range(p.nMachines):
                n += self.x[i,j].value()*j
            schedule.append(int(np.round(n)))
        return schedule, p.targetfcn(schedule)
        
    def debugPrint(self, p: Problem):
        assert self.x
        assert self.u
        
        index = [0]*p.nMachines
        for i in range(p.nMachines):
            for j in range(p.nMachines):
                index[i] += int(self.x[i,j].value()*j)
                                
        print(f"{'idx':5}| x[i,j]:")
        for i in range(p.nMachines):
            str = ""
            for j in range(p.nMachines):
                str += f"{self.x[i,j].value()} "
            print(f"{index[i]:<5}| {str}")
        print()

        print(f"{'idx':5}| u[i]:")
        for i in range(p.nMachines):
            str = ""
            for j in range(p.nMachines):
                str += f"{int(self.u[i,j].value())} "
            print(f"{index[i]:<5}| {str}")
        print()
            
        print(f"{'idx':5}| {'durat':>10} {'cur_t':>10} {'deadline':>10} is_ok")
        cur_t = 0
        for i in range(p.nMachines):
            duration = 0
            deadline = 0    
            is_ok = 0
            for j in range(p.nMachines):
                duration += self.x[i,j].value()*p.t[j]
                deadline += self.x[i,j].value()*p.d[j]
                is_ok += int(self.u[i,j].value())
            cur_t += duration
            print(f"{index[i]:<5}| {duration:10f} {cur_t:10f} {deadline:10f} ok={is_ok}")

Nsamples = 10000
np.random.seed(1)
seeds = np.random.randint(0, 1000000000, Nsamples)
bruteforceSolver = BruteforceSolver()
pulpSolver = PulpSolver()
sols = []
for n in range(Nsamples):
    print(f"seeds[{n}] = {seeds[n]}")
    np.random.seed(seeds[n])
    
    p = Problem()
    p.randomize()
    print(p)
    schedule, true_value = bruteforceSolver.solve(p)
    print(schedule)
    print(f"objective: {true_value}")

    schedule, value = pulpSolver.solve(p)
    pulpSolver.debugPrint(p)
    assert p.checkSolution(schedule)
    print(f"objective: {p.targetfcn(schedule)}")
    assert np.isclose(value, true_value, 1e-10, 1e-10)
    sols.append(true_value)
print("\n\n")

print("All tests are Ok")        
print(f"Bruteforce solver time: {bruteforceSolver.elapsed}")
print(f"PuLP solver time: {pulpSolver.elapsed}")
#print(sols)

#p = Problem()
#p.randomize(12)
#schedule, value = pulpSolver.solve(p)
#pulpSolver.debugPrint(p)

