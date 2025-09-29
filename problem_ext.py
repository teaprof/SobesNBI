from problem import Problem   
import numpy as np
import dataclasses

class DataExtractor: ...
        
@dataclasses.dataclass
class ExtendedProblem(Problem):
    # constaints: if befores contains pair (i,j) then product i should be processed before product j
    befores: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    # fine: if product j is finished after the its deadline d[j] is expired then fine fines[j] is applied
    fines: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    
    #def __init__(self, *args, **kwargs):
    #    Problem.__init__(self, *args, **kwargs)
        
    def randomize(self, nProducts=None):
        if nProducts is None:
            self.nProducts = np.random.randint(1,10)
        else:
            self.nProducts = nProducts        
        Problem.randomize(self, self.nProducts)
        self.fines = np.random.rand(self.nProducts)
        perm = list(range(self.nProducts))
        np.random.shuffle(perm)
        while len(perm) > 1:
            idx1 = np.random.randint(0, len(perm))
            idx2 = np.random.randint(0, len(perm))
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            self.befores.append((perm[idx1], perm[idx2]))
            perm = perm[:idx1] + perm[idx1+1:idx2] + perm[idx2+1:]
           
    def objectiveFcn(self, schedule: list[int]) -> float:
        assert len(schedule) == self.nProducts
        assert len(set(schedule)) == self.nProducts
        tend = np.cumsum(self.t[schedule])
        bonus = 0
        for n in range(self.nProducts):
            if tend[n] < self.d[schedule[n]]:
                bonus += self.p[schedule[n]]
            else:
                bonus -= self.fines[schedule[n]]
        return bonus
    
    # This function by the bruteforce solver
    def checkSchedule(self, schedule: list[int]) -> bool:        
        if(not super().checkSchedule(schedule)):
            return False
        # Check before-like constraints
        for n1, n2 in self.befores:
            ord1 = schedule.index(n1)
            ord2 = schedule.index(n2)
            if ord1 > ord2:
                return False
        return True
