from problem import Problem
from solver import PulpSolver
import numpy as np
import pytest

def test_problem_single_product_a():
    p = Problem(1)
    p.d = np.asarray([0])
    p.t = np.asarray([1])
    p.p = np.asarray([1])
    solver = PulpSolver()
    schedule, reward = solver.solve(p)
    assert reward == 0
    assert schedule == [0]
    
def test_problem_single_product_b():
    p = Problem(1)
    p.d = np.asarray([2])
    p.t = np.asarray([1])
    p.p = np.asarray([1])
    solver = PulpSolver()
    schedule, reward = solver.solve(p)
    assert reward == 1
    assert schedule == [0]
    
def test_problem_many_products_a():
    p = Problem(3)
    p.d = np.asarray([0.1, 0.1, 0.1])
    p.t = np.asarray([1, 1, 1])
    p.p = np.asarray([1, 1, 1])
    solver = PulpSolver()
    schedule, reward = solver.solve(p)
    assert reward == 0
    assert set(schedule) == set([0, 1, 2])
    
def test_problem_many_products_b():
    p = Problem(3)
    p.d = np.asarray([1.1, 2.1, 3.1])
    p.t = np.asarray([1, 1, 1])
    p.p = np.asarray([1, 2, 4])
    solver = PulpSolver()
    schedule, reward = solver.solve(p)
    assert reward == 7
    assert schedule == [0, 1, 2]
    
def test_problem_many_products_c():
    p = Problem(3)
    p.d = np.asarray([0.1, 1.1, 2.1])
    p.t = np.asarray([1, 1, 1])
    p.p = np.asarray([1, 2, 4])
    solver = PulpSolver()
    schedule, reward = solver.solve(p)
    assert reward == 6
    assert schedule == [1, 2, 0]
        