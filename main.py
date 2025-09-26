import pulp
import numpy as np
import dataclasses
import itertools


@dataclasses.dataclass
class Problem:
    nMachines: int
    t: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    d: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    p: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    
    def randomize(self):
        self.t = np.random.rand(self.nMachines)
        self.d = np.random.rand(self.nMachines)*self.nMachines
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
        
        
def bruteForceSolve(p: Problem): #-> tuple(list[int], float):
    schedule0 = list(range(p.nMachines))
    best_schedule = schedule0
    best_value = p.targetfcn(schedule0)
    for schedule in itertools.permutations(schedule0):
        cur_value = p.targetfcn(schedule)
        if cur_value > best_value:
            best_value = cur_value
            best_schedule = schedule
    return best_schedule, best_value
            
    
        


p = Problem(6)
p.randomize()
print(p)
solution, value = bruteForceSolve(p)
print(solution)
print(value)
    