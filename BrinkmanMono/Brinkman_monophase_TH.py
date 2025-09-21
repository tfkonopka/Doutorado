from fenics import *
import matplotlib.pyplot as plt
import time


class left_boundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-14
        return on_boundary and abs(x[0]) < tol


class right_boundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-14
        return on_boundary and abs(x[0] - 1) < tol


class bottom_boundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-14
        return on_boundary and abs(x[1]) < tol


class upper_boundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-14
        return on_boundary and abs(x[1] - 1) < tol


class Obstacle(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (0.4, 0.6)) and between(x[0], (0.3, 0.5))


mesh = UnitSquareMesh(50, 50, "crossed")

# Define function spaces and trial/test functions
# V = FiniteElement("RT", mesh.ufl_cell(), 1)  # velocity space
# Q = FiniteElement("DG", mesh.ufl_cell(), 0)  # pressure space
V = VectorElement("P", mesh.ufl_cell(), 2)  # velocity element
Q = FiniteElement("P", mesh.ufl_cell(), 1)  # pressure element


# P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
# B = FiniteElement("Bubble", mesh.ufl_cell(), mesh.topology().dim() + 1)
# V = VectorElement(NodalEnrichedElement(P1, B))
# Q = P1
# W = FunctionSpace(mesh, V * Q)

start_time = time.time()

Element = V * Q
W = FunctionSpace(mesh, Element)

(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

Gamma_Left = left_boundary()
Gamma_Right = right_boundary()
Gamma_Bottom = bottom_boundary()
Gamma_Upper = upper_boundary()

Gamma_Left.mark(boundary_parts, 0)
Gamma_Right.mark(boundary_parts, 1)
Gamma_Bottom.mark(boundary_parts, 2)
Gamma_Upper.mark(boundary_parts, 3)


obstacle = Obstacle()
domains = MeshFunction("size_t", mesh, mesh.topology().dim())
domains.set_all(0)
obstacle.mark(domains, 1)

# bc1 = DirichletBC(W.sub(0).sub(0), Constant(0.0), boundary_parts, 0)
# bc2 = DirichletBC(W.sub(0).sub(0), Constant(0.0), boundary_parts, 1)
bc3 = DirichletBC(W.sub(0).sub(1), Constant(0.0), boundary_parts, 2)
bc4 = DirichletBC(W.sub(0).sub(1), Constant(0.0), boundary_parts, 3)

bcs = [bc3, bc4]

ds = Measure("ds", domain=mesh, subdomain_data=boundary_parts)
dx = Measure("dx", domain=mesh, subdomain_data=domains)


mu = Constant(0.001002)
mu_b = Constant(10.02)
k = Constant(1)
pin = Constant(6000)
pout = Constant(0)

f = Constant((0.0, 0.0))
n = FacetNormal(mesh)


a = (
    mu_b * inner(grad(u), grad(v)) * dx(1)
    + mu_b * inner(grad(u), grad(v)) * dx(0)
    + mu / k * inner(u, v) * dx(0)
    + mu / k * inner(u, v) * dx(1)
    - div(v) * p * dx(1)
    - div(u) * q * dx(0)
    - div(v) * p * dx(0)
    - div(u) * q * dx(1)
)


L = (
    inner(f, v) * dx(0)
    + inner(f, v) * dx(1)
    - pin * dot(v, n) * ds(0)
    - pout * dot(v, n) * ds(1)
)

U = Function(W)

solve(a == L, U, bcs)

(u1, p1) = U.split()


u_file = File("/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/Mono_TH_4/u.pvd")
u_file << u1
p_file = File("/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/Mono_TH_4/p.pvd")
p_file << p1


print("---%s  seconds--" % (time.time() - start_time))
plt.show()
