"Versão com multiplos vugg sintéticos. dt fixo em 200s. "
# -*- coding: utf-8 -*-

from fenics import *
import time
import ufl_legacy as ufl
import os


class PiecewiseConstant(UserExpression):
    def __init__(self, values, markers, **kwargs):
        self._values = values
        self._markers = markers
        super().__init__(**kwargs)

    def eval_cell(self, values, x, cell):
        values[0] = self._values[self._markers[cell.index]]

    def value_shape(self):
        return tuple()


def tensor_jump(v, n):
    return ufl.outer(v, n)("+") + ufl.outer(v, n)("-")


# Fractional flow function
def F(s, mu_rel, no, nw):
    return s**nw / (s**nw + mu_rel * (1.0 - s) ** no)


def BrinkmanIMPES(Nx, _folder_base, mu_w, mu_o, perm_darcy, dt, pin, pout):
    Ny = Nx

    dir1 = _folder_base + "/dir1"
    dir2 = _folder_base + "/dir2"

    try:
        os.mkdir(dir1)
    except OSError as error:
        print(error)

    try:
        os.mkdir(dir2)
    except OSError as error:
        print(error)

    mu = (mu_o + mu_w) / 2  # Pa.s
    mu_b = (mu_o + mu_w) / 2  # Pa.s

    mili_darcy = 9.86923e-16  # Porous media permeability convertion factor md [m2] (1 Darcy = E-12 m2)

    k_matriz = perm_darcy * mili_darcy

    kgf_cm2_to_Pa = 98066.5
    pin = pin * kgf_cm2_to_Pa  # Pa
    pout = pout * kgf_cm2_to_Pa  # Pa

    dt = Constant(dt)  # s

    phi = 0.2
    mu_rel = mu_w / mu_o

    sbar = Constant(1)

    Kinv = Constant(1 / k_matriz)

    mu = Constant(mu)
    mu_b = Constant(mu_b)
    t = 0

    mesh = UnitSquareMesh(Nx, Ny, "crossed")
    # mesh = IntervalMesh(10, 0, 1)
    order = 1
    V = FiniteElement("BDM", mesh.ufl_cell(), order)
    R1 = FiniteElement("DG", mesh.ufl_cell(), order - 1)

    W = FunctionSpace(mesh, V)
    R = FunctionSpace(mesh, R1)

    u = TrialFunctions(W)
    v = TestFunctions(W)

    s = TrialFunction(R)
    r = TestFunction(R)

    U = Function(W)
    S = Function(R)
    s0 = Function(R)

    s0.vector()[:] = 0.0

    no = 2
    nw = 2

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

    bc1 = DirichletBC(W, Constant((1.0e-6, 0.0)), boundaries, 1)
    bc2 = DirichletBC(W, Constant((0.0, 0.0)), boundaries, 2)
    # # # bc3 = DirichletBC(VQ.sub(0), Constant((0.0, 0.0)), boundaries, 3)
    bc4 = DirichletBC(W, Constant((0.0, 0.0)), boundaries, 4)

    bcs = [bc1, bc2, bc4]  # velocity BC

    # Juntando as condições de contorno

    alpha = 35
    h = CellDiameter(mesh)
    h2 = ufl.Min(h("+"), h("-"))

    f = Constant((0.0, 0.0))
    n = FacetNormal(mesh)

    stab = (
        mu * (alpha / h2) * inner(tensor_jump(u, n), tensor_jump(v, n)) * dS
        - mu * inner(avg(grad(u)), tensor_jump(v, n)) * dS
        - mu * inner(avg(grad(v)), tensor_jump(u, n)) * dS
    )

    mu_b = 1
    a = (
        mu_b * inner(grad(u), grad(v)) * dx
        + inner(v, u) * dx
        - F(s0, mu_rel, no, nw) * v * dx
        + stab
    )

    L = inner(f, v) * dx

    un = 0.5 * (inner(u, n) + abs(inner(u, n)))
    un_h = 0.5 * (inner(u, n) - abs(inner(u, n)))

    stabilisation = dt("+") * inner(jump(r), jump(un * F(s0, mu_rel, no, nw))) * dS

    L3 = (
        r * (s - s0) * dx
        - dt * inner(grad(r), *u) * dx
        + dt * r * un * ds
        + stabilisation
        + dt * r * un_h * sbar * ds(1)
    )

    a_s, L_f = lhs(L3), rhs(L3)

    u_file = XDMFFile(dir2 + "/velocity.xdmf")

    s_file = XDMFFile(dir2 + "/saturation.xdmf")

    step = 0

    # while t < T:
    while step < 10:
        # ===

        t += float(dt)
        solve(a == L, U, bcs)
        solve(a_s == L_f, S)
        s0.assign(S)

        s_file.write(S, t)
        u_file.write(u, t)


Nx = 20
mu_w = 0.01
mu_o = 0.01
perm_darcy = 100
dt = 800

pin = 2
pout = 1

_folder_base = [
    "/home/tfk/Desktop/results/Coclite",
    #     "/home/tfkonopka/results/Central",
    #     "/home/tfkonopka/results/vugg_4",
    #     "/home/tfkonopka/results/vugg_16",
]


for i in _folder_base:
    try:
        os.mkdir(i)
    except OSError as error:
        print(error)

t_total_i = start_time = time.time()
BrinkmanIMPES(Nx, _folder_base[0], mu_w, mu_o, perm_darcy, dt, pin, pout)
t_total = time.time() - t_total_i

print(f"Tempo Total de simulações = {t_total}")
