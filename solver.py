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
        # some big numbers used to convert Heaviside step functions to linear ones
        self.M = None
        self.M2 = None
        # some helper variable
        self.all_indices = None
        #time elapsed
        self.elapsed = 0
        
    @staticmethod
    def addStepFunction(problem: pulp.LpProblem, x, u, M):
        """ Add step function conditions 
                if x >= 0 then u = 1
                if x < 0 then u = 0         
            to the problem. This is implemented by means of two additional constraints:
                -(1-u)*M < x <= M*u,
            where M is some positive constant such that M > sup(abs(x)).
            Args:
                u: logical variable
        """
        problem += x - M*u <= 0
        problem += x + M*(1-u) >= 0
        
        
    def buildPulpProblem(self, p: Problem) -> pulp.LpProblem:
        """ converts `p` to pulp problem.
        This method can be redefined in derived classes to add extra constrains
        
        Args:
            p: Problem

        Returns:
            pulp.LpProblem:
        """
        problem = pulp.LpProblem("Scheduler problem", pulp.LpMaximize)
        
        # encode the schedule with bit matrix x[i, j]
        # i - the number of step in the schedule, j - the number of product
        self.all_indices = [(i,j) for i in range(p.nProducts) for j in range(p.nProducts)]
        self.x = pulp.LpVariable.dicts("x", self.all_indices, cat=pulp.const.LpBinary)
        
        # each row and each column should have exactlty one `1`
        for i in range(p.nProducts):
                problem += pulp.lpSum([self.x[i,j] for j in range(p.nProducts)]) == 1
                problem += pulp.lpSum([self.x[j,i] for j in range(p.nProducts)]) == 1
                
        
        # Define matrix `u` which encodes which rewards are obtained for specified schedule x:
        #       u[i, j] = step(deadline[j] - tfinish[j])*x[i,j],
        # where step(x) is Heaviside step function.
           
        t_finish = 0 # time when current operation is finished
        # some magic formulas, we can't write here arbitrary big values because it can lead 
        # to a loss of precision (ints becomes fractionals)
        self.M2 = 2*np.max(p.d)
        self.M = 2*(np.sum(p.t) + np.max(p.d))
        # declare and initialize u
        self.u = pulp.LpVariable.dicts("u", self.all_indices, cat = pulp.const.LpBinary)
        for i, j in self.all_indices:
                duration = self.x[i,j]*p.t[j]
                t_finish = t_finish + duration
                deadline = p.d[j] - self.M2*(1-self.x[i,j]) # deadline is negative is x[i,j] = 0 (=> reward is missed)
                #problem += deadline - t_finish - self.M*self.u[i,j] <= 0
                #problem += deadline - t_finish + self.M*(1-self.u[i,j]) >= 0
                PulpSolver.addStepFunction(problem, deadline - t_finish, self.u[i,j], self.M)

        # objective function is the sum of all rewards
        problem += pulp.lpSum([self.u[i,j]*p.p[j] for i, j in self.all_indices])    
        return problem

        
    
    def solve(self, p: Problem) -> tuple[list[int], float]:
        """ solve problem `p` using pulp library.
        
        Args:
            p: problem

        Returns:
            tuple[list[int], float]: schedule and the reached reward (max value of the objective function)
        """
        
        # build problem (convert `problem` to pulp.lpProblem)
        self.problem = self.buildPulpProblem(p)
        
        # solve, round and accumulate time elapsed
        self.problem.solve()
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
        """ pretty printing some debug information

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
