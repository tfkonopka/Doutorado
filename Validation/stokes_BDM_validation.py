from dolfin import *
import ufl
import numpy as np
import matplotlib.pyplot as plt

Nx = 50
Ny = 400
# Load mesh and subdomains
mesh = UnitSquareMesh(Nx, Ny, "crossed")
n = FacetNormal(mesh)

order = 1
V = FiniteElement("BDM", mesh.ufl_cell(), order)
Q = FiniteElement("DG", mesh.ufl_cell(), order - 1)
Element = MixedElement([V, Q])
W = FunctionSpace(mesh, Element)


boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
left = AutoSubDomain(lambda x: near(x[0], 0.0))
right = AutoSubDomain(lambda x: near(x[0], 1.0))
bottom = AutoSubDomain(lambda x: near(x[1], 0.0))
top = AutoSubDomain(lambda x: near(x[1], 1.0))

# Define boundary markers
left.mark(boundaries, 1)
top.mark(boundaries, 2)
right.mark(boundaries, 3)
bottom.mark(boundaries, 4)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)


bc2 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 2)
bc4 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 4)


# Collect boundary conditions
bcs = [bc2, bc4]

# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
f = Constant((0, 0))
mu = 1
pin = 10  # 1.0
pout = 0.0

alpha = 35
h = CellDiameter(mesh)
h2 = ufl.Min(h("+"), h("-"))


def tensor_jump(v, n):
    return ufl.outer(v, n)("+") + ufl.outer(v, n)("-")


stab = (
    mu * (alpha / h2) * inner(tensor_jump(u, n), tensor_jump(v, n)) * dS
    - mu * inner(avg(grad(u)), tensor_jump(v, n)) * dS
    - mu * inner(avg(grad(v)), tensor_jump(u, n)) * dS
)

nitche = (
    alpha / h * inner(outer(v, n), outer(u, n)) * ds(2)
    - inner(grad(u), outer(v, n)) * ds(2)
    - inner(grad(v), outer(u, n)) * ds(2)
    + alpha / h * inner(outer(v, n), outer(u, n)) * ds(4)
    - inner(grad(u), outer(v, n)) * ds(4)
    - inner(grad(v), outer(u, n)) * ds(4)
)

a = (
    mu * inner(grad(u), grad(v)) * dx
    - div(v) * p * dx
    + div(u) * q * dx
    + stab
    + nitche
)
L = inner(f, v) * dx - pin * dot(v, n) * ds(1) - pout * dot(v, n) * ds(3)


# Compute solution
w = Function(W)
solve(a == L, w, bcs)

# Split the mixed solution using deepcopy
# (needed for further computation on coefficient vector)
(u, p) = w.split(True)

# Save solution in VTK format
ufile_pvd = File("/home/tfk/Desktop/results/Validation/Stokes/Stokes_mono_BDM/u.pvd")
ufile_pvd << u
pfile_pvd = File("/home/tfk/Desktop/results/Validation/Stokes/Stokes_mono_BDM/p.pvd")
pfile_pvd << p
