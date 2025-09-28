import lark
import pulp
from problem import Problem
from solver import PulpSolver
from crash_test import BruteforceSolver
import numpy as np

grammar = """
start: products befores fines
products: product *
product: "product" NAME "=" FLOAT "," FLOAT  "," FLOAT
befores: before *
before: NAME "before" NAME
fines: fine *
fine: "fine for" NAME "is" FLOAT

FLOAT: /[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?/        
NAME: /[a-zA-Z][a-zA-Z0-9]*/
WHITESPACE: (" " | "\\n")+
COMMENT: "#" /[^\\n]/*
%ignore WHITESPACE
%ignore COMMENT
"""

class DataExtractor(lark.Transformer):
    """Extract data from AST created by lark parser"""
    def __init__(self) : 
        lark.Transformer.__init__(self)
        self.d = dict()
        self.t = dict()
        self.p = dict()
        self.f = dict()        
        self.before_list = []
        self.product_names = []
        
    def before(self, tree):
        # the name of this method must correspond to the name of grammar rule
        name1 = tree[0].value
        name2 = tree[1].value
        self.before_list.append((name1, name2))
        
        
    def product(self, tree):
        # the name of this method must correspond to the name of grammar rule
        name = tree[0].value
        self.t[name] = float(tree[1].value)
        self.d[name] = float(tree[2].value)
        self.p[name] = float(tree[3].value)
        self.product_names.append(name)
        
    
    def fine(self, tree):
        # the name of this method must correspond to the name of grammar rule
        name = tree[0].value
        fine = float(tree[1].value)
        self.f[name] = fine
        
class ExtendedProblem(Problem):
    def __init__(self, *args, **kwargs):
        Problem.__init__(self, *args, **kwargs)
        self.befores = []
        self.fines = []
        
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
        
    @staticmethod
    def initializeByParsedTree(tree):
        extractor = DataExtractor()
        extractor.transform(tree)
        nProducts = len(extractor.d)
        res = ExtendedProblem(nProducts, np.zeros(nProducts), np.zeros(nProducts), np.zeros(nProducts))
        for name, t in extractor.t.items():
            n = extractor.product_names.index(name)
            res.t[n] = np.asarray(t)
        for name, d in extractor.d.items():
            n = extractor.product_names.index(name)
            res.d[n] = np.asarray(d)
        for name, p in extractor.p.items():
            n = extractor.product_names.index(name)
            res.p[n] = np.asarray(p)
        for name1, name2 in extractor.before_list:
            n1 = extractor.product_names.index(name1)
            n2 = extractor.product_names.index(name2)
            res.befores.append((n1, n2))
        res.fines = np.zeros(nProducts)
        for name, fine in extractor.f.items():
            n = extractor.product_names.index(name)
            res.fines[n] = fine
        return res                
        
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

        

if __name__ == '__main__':
    # Read file
    filename = "problem.schedlang"
    with open(filename, "r") as f:
        description = f.read()

    # Parse strings and build AST
    parser = lark.Lark(grammar)
    tree = parser.parse(description)
    print(tree)
        
    # Create ExtendedProblem
    ext_problem = ExtendedProblem.initializeByParsedTree(tree)
    
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
