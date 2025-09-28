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

filename = "problem.schedlang"
with open(filename, "r") as f:
    description = f.read()

parser = lark.Lark(grammar)
tree = parser.parse(description)
print(tree)

class Converter(lark.Transformer):
    def __init__(self) : 
        lark.Transformer.__init__(self)
        self.d = dict()
        self.t = dict()
        self.p = dict()
        self.f = dict()        
        self.before_list = []
        self.product_names = []
        
    def before(self, tree):
        name1 = tree[0].value
        name2 = tree[1].value
        self.before_list.append((name1, name2))
        
        
    def product(self, tree):
        name = tree[0].value
        self.t[name] = float(tree[1].value)
        self.d[name] = float(tree[2].value)
        self.p[name] = float(tree[3].value)
        self.product_names.append(name)
        
    
    def fine(self, tree):
        name = tree[0].value
        fine = float(tree[1].value)
        self.f[name] = fine
        
converter = Converter()
tt = converter.transform(tree)
print("\n")
print(converter.f)
print(converter.p)


class ExtendedProblem(Problem):
    def __init__(self, *args, **kwargs):
        Problem.__init__(self, *args, **kwargs)
        self.befores = []
        self.fines = []
        
    @staticmethod
    def initializeByParsedTree(tree):
        converter = Converter()
        converter.transform(tree)
        nProducts = len(converter.d)
        res = ExtendedProblem(nProducts, np.zeros(nProducts), np.zeros(nProducts), np.zeros(nProducts))
        for name, t in converter.t.items():
            n = converter.product_names.index(name)
            res.t[n] = np.asarray(t)
        for name, d in converter.d.items():
            n = converter.product_names.index(name)
            res.d[n] = np.asarray(d)
        for name, p in converter.p.items():
            n = converter.product_names.index(name)
            res.p[n] = np.asarray(p)
        for name1, name2 in converter.before_list:
            n1 = converter.product_names.index(name1)
            n2 = converter.product_names.index(name2)
            res.befores.append((n1, n2))
        res.fines = np.zeros(nProducts)
        for name, fine in converter.f.items():
            n = converter.product_names.index(name)
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
    
    def checkSolution(self, schedule: list[int]) -> bool:
        if(not super().checkSolution(schedule)):
            return False
        for n1, n2 in self.befores:
            ord1 = schedule.index(n1)
            ord2 = schedule.index(n2)
            if ord1 > ord2:
                return False
        return True
        
        
ext_problem = ExtendedProblem.initializeByParsedTree(tree)



class ExtendedPulpSolver(PulpSolver):
    def buildPulpProblem(self, p: ExtendedProblem) -> pulp.LpProblem:
        problem = PulpSolver.buildPulpProblem(self, p)
        befores = pulp.LpVariable.dicts("befores", p.befores)
        for n1, n2 in p.befores:
            n1_ord = pulp.lpSum([self.x[i, n1]*i for i in range(p.nProducts)])
            n2_ord = pulp.lpSum([self.x[i, n2]*i for i in range(p.nProducts)])
            problem += (n1_ord - n2_ord <= 0.0)
            
        t_finish = 0 # time when current operation is finished
        # declare and initialize v:
        self.v = pulp.LpVariable.dicts("v", self.all_indices, cat = pulp.const.LpBinary)
        for i, j in self.all_indices:
                duration = self.x[i,j]*p.t[j]
                t_finish = t_finish + duration
                deadline = p.d[j] + self.M2*(1-self.x[i,j])*10 # deadline is big positive number if x[i,j] = 0 (no fine is applied)
                PulpSolver.addStepFunction(problem, deadline - t_finish, self.v[i,j], self.M2*100)

        # objective function is the sum of all rewards
        problem += pulp.lpSum([self.u[i,j]*p.p[j] for i, j in self.all_indices]) - pulp.lpSum([(1-self.v[i,j])*p.fines[j] for i, j in self.all_indices])
        
        return problem
    
    def debugPrint(self, p: Problem):
        super().debugPrint(p)
        print(f"{'idx':5}| v[i]:")
        for i in range(p.nProducts):
            str = ""
            for j in range(p.nProducts):
                str += f"{int(self.v[i,j].value())} "
            print(f"{self.schedule[i]:<5}| {str}")
        print()

    
    
    
    
solver = ExtendedPulpSolver()
schedule, reward = solver.solve(ext_problem)
assert ext_problem.checkSolution(schedule)
print("Schedule (solution):")
print(schedule)
print(f"Reward {reward} out of {np.sum(ext_problem.p)}")

bruteforceSolver = BruteforceSolver()
_, reward0 = bruteforceSolver.solve(ext_problem)

print(f"Bruteforce solution reward: {reward0}")

assert np.isclose(reward, reward0, 1e-5)
print("All Ok")
