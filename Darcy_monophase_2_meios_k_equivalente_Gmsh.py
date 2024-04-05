from fenics import *
import matplotlib.pyplot as plt
import time
import os
import numpy as np


def perm_2k(perm_m, perm_v, caminho):
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

    mesh = Mesh()
    with XDMFFile(caminho + "/mesh/mesh.xdmf") as infile:
        infile.read(mesh)

    mvc = MeshValueCollection("size_t", mesh, 2)
    with XDMFFile(caminho + "/mesh/domains.xdmf") as infile:
        infile.read(mvc)
    Markers = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    mvc = MeshValueCollection("size_t", mesh, 1)
    with XDMFFile(caminho + "/mesh/boundaries.xdmf") as infile:
        infile.read(mvc)
    boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    order = 1
    V = FiniteElement("BDM", mesh.ufl_cell(), order)
    Q = FiniteElement("DG", mesh.ufl_cell(), order - 1)

    Element = V * Q
    W = FunctionSpace(mesh, Element)

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    # bc1 = DirichletBC(W.sub(0), Constant((1.0e-6, 0.0)), boundaries, 1)
    # bc1 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 1)  # face Oeste
    bc2 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 2)  # face Norte
    # bc3 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 3)  # face leste2.2.6
    bc4 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 4)  # face sul

    bcs = [bc2, bc4]  # velocity BC

    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    dx = Measure("dx", domain=mesh, subdomain_data=Markers)

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
    "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/vugg_sintetico/Mono_Brinkman/DR_Perm_1mD",
    "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/vugg_sintetico/Mono_Brinkman/DR_Perm_10mD",
    "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/vugg_sintetico/Mono_Brinkman/DR_Perm_100mD",
    "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/vugg_sintetico/Mono_Brinkman/DR_Perm_1000mD",
]


caso1 = "/array_perm_darcy"
perm_brinkman = [
    1.5401588753367232,
    15.401588753367232,
    154.01588753367232,
    1540.1588753367232,
]  # CANAL
perm_matrix = [1, 10, 100, 1000]
num_values = 500

lim_inf = [1, 10, 100, 1000]
lim_sup = [1e6, 1e7, 1e8, 1e9]

k = []
erro = []

for j in range(len(_folder_base)):
    perm_vugg = np.logspace(np.log10(lim_inf[j]), np.log10(lim_sup[j]), num_values)
    k = []
    erro = []
    for i in range(len(perm_vugg)):
        x = perm_2k(perm_matrix[j], perm_vugg[i], _folder_base[j])
        k.append(x)
        _erro = np.abs((perm_brinkman[j] - x) / perm_brinkman[j])
        erro.append(_erro)

    DataRecord(k, perm_vugg, erro, perm_brinkman[j], caso1, _folder_base[j])
