from problem import Problem
import pulp
import numpy as np

class PulpSolver:
    def __init__(self):
        self.problem = None
        # x[i, j] is 1 if on the i-th step of the schedule the j-th product is processed, otherwise 0
        self.x = None        
        # u[i, j] is 1 if on the i-th step the j-th product is produced and finished before the deadline, otherwise 0
        self.u = None
        # some helper variable
        self.all_indices = None
        #time elapsed
        self.elapsed = 0
        
    @staticmethod
    def addStepFunctionConstrains(problem: pulp.LpProblem, x: pulp.LpVariable, u: pulp.LpVariable, M: float):
        """ Add constraints to `problem` such that `u` becomes equal
        to h(x), where h(x) is a Heaviside step function defined as:
                h(x) = 1, if x >=0
                h(x) = 0, otherwise
        The conditions which makes u equal to h(x) have a form:
                -(1-u)*M < x <= M*u,
        where M is some positive constant such that M > sup(abs(x)) + eps.
        Args:
            x: the variable which value should be checked
            u: auxiliary logical variable
            M: some big constant (see above)
        """
        problem += x - M*u <= 0
        problem += x + M*(1-u) >= 0
        
        
    def initializePulpProblem(self, p: Problem):
        """ converts `p` to pulp problem.
        This method can be redefined in derived classes to add extra constrains
        
        Args:
            p: Problem

        Returns:
            pulp.LpProblem:
        """
        self.problem = pulp.LpProblem("Scheduler problem", pulp.LpMaximize)
        
        # encode the schedule with bit matrix x[i, j]
        # i - the number of step in the schedule, j - the number of product
        self.all_indices = [(i,j) for i in range(p.nProducts) for j in range(p.nProducts)]
        self.x = pulp.LpVariable.dicts("x", self.all_indices, cat=pulp.const.LpBinary)
        
        # each row and each column should have exactlty one `1`
        for i in range(p.nProducts):
                self.problem += pulp.lpSum([self.x[i,j] for j in range(p.nProducts)]) == 1
                self.problem += pulp.lpSum([self.x[j,i] for j in range(p.nProducts)]) == 1
                
        
        # Define matrix `u` which encodes which rewards are obtained for specified schedule x:
        #       u[i, j] = step(deadline[j] - tfinish[j])*x[i,j],
        # where step(x) is Heaviside step function.
           
        t_finish = 0 # time when current operation is finished
        # We can't write here arbitrary big values because this leads to a loss of precision (ints becomes fractionals),
        # so we estimate supremums and add eps
        eps = 0.1
        M2 = np.max(p.d) + eps
        M = (np.sum(p.t) + np.max(p.d)) + eps
        # declare and initialize u
        self.u = pulp.LpVariable.dicts("u", self.all_indices, cat = pulp.const.LpBinary)
        for i, j in self.all_indices:
                duration = self.x[i,j]*p.t[j]
                t_finish = t_finish + duration
                deadline = p.d[j] - M2*(1-self.x[i,j]) # deadline is negative is x[i,j] = 0 (=> reward is missed)
                PulpSolver.addStepFunctionConstrains(self.problem, deadline - t_finish, self.u[i,j], M)

        # objective function is the sum of all rewards
        self.problem += pulp.lpSum([self.u[i,j]*p.p[j] for i, j in self.all_indices])    

    
    def solve(self, p: Problem) -> tuple[list[int], float]:
        """ solve problem `p` using the pulp-based mixed integer linear solver.
        
        Args:
            p: problem

        Returns:
            tuple[list[int], float]: optimal schedule and the corresponding reward
        """
        
        # convert `problem` to pulp.lpProblem
        self.initializePulpProblem(p)
        
        # solve, round and accumulate time elapsed
        solver = pulp.getSolver('GUROBI')
        #solver = None # default solver
        self.problem.solve(solver)
        self.problem.roundSolution()
        self.elapsed += self.problem.solutionTime
        
        # decode x to the list of ints
        schedule: list[int] = []
        for i in range(p.nProducts):
            n = 0
            for j in range(p.nProducts):
                n += self.x[i,j].value()*j
            schedule.append(int(np.round(n)))
        
        self.schedule = schedule
        self.debugPrint(p)
                    
        reward = p.objectiveFcn(schedule)
        return schedule, reward
            
    def debugPrint(self, p: Problem):
        """ pretty print some debug information

        Args:
            p: problem
        """
        assert self.x
        assert self.u
                                        
        print(f"{'idx':5}| x[i,j]:")
        for i in range(p.nProducts):
            str = ""
            for j in range(p.nProducts):
                str += f"{self.x[i,j].value()} "
            print(f"{self.schedule[i]:<5}| {str}")
        print()

        print(f"{'idx':5}| u[i]:")
        for i in range(p.nProducts):
            str = ""
            for j in range(p.nProducts):
                str += f"{int(self.u[i,j].value())} "
            print(f"{self.schedule[i]:<5}| {str}")
        print()
            
        print(f"{'idx':5}| {'durat':>10} {'cur_t':>10} {'deadline':>10} is_ok")
        cur_t = 0
        for i in range(p.nProducts):
            duration = 0
            deadline = 0    
            is_ok = 0
            for j in range(p.nProducts):
                duration += self.x[i,j].value()*p.t[j]
                deadline += self.x[i,j].value()*p.d[j]
                is_ok += int(self.u[i,j].value())
            cur_t += duration
            print(f"{self.schedule[i]:<5}| {duration:10f} {cur_t:10f} {deadline:10f} ok={is_ok}")
