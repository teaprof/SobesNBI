from solver import PulpSolver
from problem_ext import ExtendedProblem
from problem import Problem
import numpy as np
import pulp

    
class ExtendedPulpSolver(PulpSolver):
    """ Extend PulpSolver class with
            - additional constraints like "productA should be processed before productB"
            - fines if the deadline is missed    
    """
    
    def initializePulpProblem(self, p: ExtendedProblem) -> pulp.LpProblem:
        PulpSolver.initializePulpProblem(self, p)
        befores = pulp.LpVariable.dicts("befores", p.befores)
        for n1, n2 in p.befores:
            n1_ord = pulp.lpSum([self.x[i, n1]*i for i in range(p.nProducts)])
            n2_ord = pulp.lpSum([self.x[i, n2]*i for i in range(p.nProducts)])
            self.problem += (n1_ord - n2_ord <= 0.0)
            
        t_finish = 0 # time when current operation is finished
        # declare and initialize v:
        self.v = pulp.LpVariable.dicts("v", self.all_indices, cat = pulp.const.LpBinary)
        eps = 1
        M = np.sum(p.t) + eps # total time to finish all tasks
        M2 = M + np.max(p.d) + eps
        for i, j in self.all_indices:
                duration = self.x[i,j]*p.t[j]
                t_finish = t_finish + duration
                deadline = p.d[j] + M*(1-self.x[i,j]) # if x[i, j] = 0 deadline becomes a big positive number and no fine is applied
                PulpSolver.addStepFunctionConstrains(self.problem, deadline - t_finish, self.v[i,j], M2)

        # objective function is the sum of all rewards
        self.problem += pulp.lpSum([self.u[i,j]*p.p[j] for i, j in self.all_indices]) - pulp.lpSum([(1-self.v[i,j])*p.fines[j] for i, j in self.all_indices])
        
    
    def debugPrint(self, p: Problem):
        super().debugPrint(p)
        print(f"{'idx':5}| v[i]:")
        for i in range(p.nProducts):
            str = ""
            for j in range(p.nProducts):
                str += f"{int(self.v[i,j].value())} "
            print(f"{self.schedule[i]:<5}| {str}")
        print()

        
