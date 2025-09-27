import pulp

x = pulp.LpVariable('x')
u = pulp.LpVariable('u', cat=pulp.const.LpBinary)
M = 1000000
threshold = 10

p = pulp.LpProblem('problemmmm')


p += pulp.lpSum([-u])
p += x - threshold <= M*u
p += x - threshold >= -M*(1-u)

p.solve()
print(x.value())
