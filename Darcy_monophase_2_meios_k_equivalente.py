from fenics import *
import matplotlib.pyplot as plt
import time
import os


import numpy as np


class Obstacle(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (0.4, 0.6)) and between(x[0], (0.1, 0.9))


# class Obstacle(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.2, 0.4)) and between(x[0], (0.2, 0.4))


# class Obstacle1(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.2, 0.4)) and between(x[0], (0.6, 0.8))


# class Obstacle2(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.6, 0.8)) and between(x[0], (0.6, 0.8))


# class Obstacle3(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.6, 0.8)) and between(x[0], (0.2, 0.4))


# class Obstacle(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.15, 0.25)) and between(x[0], (0.15, 0.25))


# class Obstacle1(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.15, 0.25)) and between(x[0], (0.35, 0.45))


# class Obstacle2(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.15, 0.25)) and between(x[0], (0.55, 0.65))


# class Obstacle3(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.15, 0.25)) and between(x[0], (0.75, 0.85))


# class Obstacle4(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.35, 0.45)) and between(x[0], (0.15, 0.25))


# class Obstacle5(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.35, 0.45)) and between(x[0], (0.35, 0.45))


# class Obstacle6(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.35, 0.45)) and between(x[0], (0.55, 0.65))


# class Obstacle7(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.35, 0.45)) and between(x[0], (0.75, 0.85))


# class Obstacle8(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.55, 0.65)) and between(x[0], (0.15, 0.25))


# class Obstacle9(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.55, 0.65)) and between(x[0], (0.35, 0.45))


# class Obstacle10(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.55, 0.65)) and between(x[0], (0.55, 0.65))


# class Obstacle11(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.55, 0.65)) and between(x[0], (0.75, 0.85))


# class Obstacle12(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.75, 0.85)) and between(x[0], (0.15, 0.25))


# class Obstacle13(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.75, 0.85)) and between(x[0], (0.35, 0.45))


# class Obstacle14(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.75, 0.85)) and between(x[0], (0.55, 0.65))


# class Obstacle15(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.75, 0.85)) and between(x[0], (0.75, 0.85))


def perm_2k(perm_m, perm_v):
    perm_darcy = perm_m
    perm_vugg = perm_v

    mili_darcy = 9.86923e-16  # Porous media permeability convertion factor md [m2] (1 Darcy = E-12 m2)
    k_matriz = perm_darcy * mili_darcy
    k_vuug = perm_vugg * mili_darcy
    k0 = Constant(k_matriz)
    k1 = Constant(k_vuug)

    marker_inner = 1
    marker_outer = 0

    kgf_cm2_to_Pa = 98066.5

    pin = 2 * kgf_cm2_to_Pa  # Pa
    pout = kgf_cm2_to_Pa  # Pa
    _pout = pout
    mu = Constant(0.00100)

    pin = Constant(pin)
    pout = Constant(pout)

    mesh = UnitSquareMesh(10, 10, "crossed")

    # Define function spaces and trial/test functions
    V = FiniteElement("BDM", mesh.ufl_cell(), 1)  # velocity space
    Q = FiniteElement("DG", mesh.ufl_cell(), 0)  # pressure space

    start_time = time.time()

    Element = V * Q
    W = FunctionSpace(mesh, Element)

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

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

    obstacle = Obstacle()
    # obstacle1 = Obstacle1()
    # obstacle2 = Obstacle2()
    # obstacle3 = Obstacle3()
    # obstacle4 = Obstacle4()
    # obstacle5 = Obstacle5()
    # obstacle6 = Obstacle6()
    # obstacle7 = Obstacle7()
    # obstacle8 = Obstacle8()
    # obstacle9 = Obstacle9()
    # obstacle10 = Obstacle10()
    # obstacle11 = Obstacle11()
    # obstacle12 = Obstacle12()
    # obstacle13 = Obstacle13()
    # obstacle14 = Obstacle14()
    # obstacle15 = Obstacle15()
    domains = MeshFunction("size_t", mesh, mesh.topology().dim())
    domains.set_all(0)
    obstacle.mark(domains, marker_inner)
    # obstacle1.mark(domains, marker_inner)
    # obstacle2.mark(domains, marker_inner)
    # obstacle3.mark(domains, marker_inner)
    # obstacle4.mark(domains, marker_inner)
    # obstacle5.mark(domains, marker_inner)
    # obstacle6.mark(domains, marker_inner)
    # obstacle7.mark(domains, marker_inner)
    # obstacle8.mark(domains, marker_inner)
    # obstacle9.mark(domains, marker_inner)
    # obstacle10.mark(domains, marker_inner)
    # obstacle11.mark(domains, marker_inner)
    # obstacle12.mark(domains, marker_inner)
    # obstacle13.mark(domains, marker_inner)
    # obstacle14.mark(domains, marker_inner)
    # obstacle15.mark(domains, marker_inner)

    File("domains.pvd") << domains

    # bc1 = DirichletBC(W.sub(0), Constant((1.0e-6, 0.0)), boundaries, 1)
    # bc2 = DirichletBC(W.sub(0).sub(0), Constant(0.0), boundaries, 1)
    bc3 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 2)
    bc4 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 4)
    # bc5 = DirichletBC(W.sub(1), Constant(6e8), boundaries, 1)
    # bc6 = DirichletBC(W.sub(1), Constant(0), boundaries, 1)
    bcs = [bc3, bc4]

    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    dx = Measure("dx", domain=mesh, subdomain_data=domains)

    f = Constant((0.0, 0.0))
    n = FacetNormal(mesh)

    a = (
        +mu / k0 * inner(u, v) * dx(0)
        - div(v) * p * dx(0)
        - div(u) * q * dx(0)
        + mu / k1 * inner(u, v) * dx(1)
        - div(v) * p * dx(1)
        - div(u) * q * dx(1)
        # + inner(v, p * n) * ds
    )
    # L = inner(f, v) * dx
    L = inner(f, v) * dx - pout * dot(v, n) * ds(3) - pin * dot(v, n) * ds(1)

    # L = inner(f, v) * dx - pout * dot(v, n) * ds(1)

    U = Function(W)

    solve(a == L, U, bcs)

    (u1, p1) = U.split()

    area = assemble(Constant(1.0) * ds(1))
    Q = assemble(dot(u1, n) * ds(1))
    # print(f" Q =  {Q} ")

    L = assemble(1 * ds(2))

    pin_calc = assemble(p1 * ds(1))  # pressão média na entrada
    # dp = pin_calc - pout

    dp = float(pin - _pout)
    k_medium = -float(((Q / (area * dp / L))) * (mu))

    k_equivalente = float(k_medium / mili_darcy)
    print(f"k_equivalente  = {k_equivalente}")

    # u_file = File("results/darcy_mono/darcy_mono_u.pvd")
    # u_file << u1
    # p_file = File("results/darcy_mono/darcy_mono__p.pvd")
    # p_file << p1

    # print("---%s  seconds--" % (time.time() - start_time))

    # plt.show()

    return k_equivalente


def DataRecord(k, perm_vugg, erro, perm_brinkman, caso, _folder):
    f = open(_folder + caso + ".txt", "w")
    string = (
        "perm_vugg" + "," + "perm_equivalente" + "," + "erro" + "," + "perm_brinkman"
    )
    f.write(string)
    f.write("\n")
    for i in range(len(k)):
        string = "{:.5f},{:.5f},{:.5f},{:.5f}\n".format(
            perm_vugg[i], k[i], erro[i], perm_brinkman
        )
        f.write(string)
    f.close()


_folder_base = [
    "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/vugg_sintetico/Mono_Brinkman/canal_perm1000/",
    "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/vugg_sintetico/Mono_Brinkman/canal_perm10/",
    "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/vugg_sintetico/Mono_Brinkman/canal_perm1/",
    "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/vugg_sintetico/Mono_Brinkman/canal/",
]


caso1 = "array_perm_darcy"
perm_brinkman = [
    2389.44,
    23.8944,
    2.38944,
    238.944,
]  # CANAL
perm_matrix = [1000, 10, 1, 100]
num_values = 100

lim_inf = [1000, 10, 1, 100]
lim_sup = [1e8, 1e6, 1e5, 1e7]

k = []
erro = []

for j in range(len(_folder_base)):
    perm_vugg = np.logspace(np.log10(lim_inf[j]), np.log10(lim_sup[j]), num_values)
    k = []
    erro = []
    for i in range(len(perm_vugg)):
        x = perm_2k(perm_matrix[j], perm_vugg[i])
        k.append(x)
        _erro = np.abs((perm_brinkman[j] - x) / perm_brinkman[j])
        erro.append(_erro)

    DataRecord(k, perm_vugg, erro, perm_brinkman[j], caso1, _folder_base[j])
