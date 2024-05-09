from fenics import *
import time
import ufl_legacy as ufl
import os

# from brinkman_biphase_IMPES_v2_lib import *


class Obstacle(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[0], (0.3, 0.7)) and between(x[1], (0.3, 0.7))


# class Obstacle(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.3, 0.7)) and between(x[0], (0.3, 0.7))


# class Obstacle(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.4, 0.6)) and between(x[0], (0.1, 0.9))


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
#         return between(x[1], (0.15, 0.25)) and between(x[0], (0.75, 0.85))  #


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


def tensor_jump(v, n):
    return ufl.outer(v, n)("+") + ufl.outer(v, n)("-")


def BrinkmanMonoBDM(k_matriz, pin, pout, mu, Nx, Ny, caminho, alpha):
    start_time = time.time()

    dir2 = caminho + "/monophase"
    _pout = pout

    mesh = UnitSquareMesh(Nx, Ny, "crossed")

    order = 1
    V = FiniteElement("BDM", mesh.ufl_cell(), order)
    Q = FiniteElement("DG", mesh.ufl_cell(), order - 1)

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

    marker_inner = 1
    marker_outer = 0

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

    bc1 = DirichletBC(W.sub(0), Constant((1.0e-6, 0.0)), boundaries, 1)
    # bc1 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 1)
    bc2 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 2)
    # # # bc3 = DirichletBC(VQ.sub(0), Constant((0.0, 0.0)), boundaries, 3)
    bc4 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 4)

    bcs = [bc1, bc2, bc4]  # velocity BC

    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    dx = Measure("dx", domain=mesh, subdomain_data=domains)

    mili_darcy = 9.86923e-16  # Porous media permeability convertion factor md [m2] (1 Darcy = E-12 m2)

    k_matriz2 = k_matriz * mili_darcy

    kgf_cm2_to_Pa = 98066.5

    pin = pin * kgf_cm2_to_Pa  # Pa
    pout = pout * kgf_cm2_to_Pa  # Pa

    mu = Constant(mu)
    k = Constant(k_matriz2)
    _pout = pout
    pin = Constant(pin)
    pout = Constant(pout)
    dp = pin - pout
    # alpha = 35
    h = CellDiameter(mesh)
    h2 = ufl.Min(h("+"), h("-"))

    f = Constant((0.0, 0.0))
    n = FacetNormal(mesh)

    stab = (
        mu * (alpha / h2) * inner(tensor_jump(u, n), tensor_jump(v, n)) * dS
        - mu * inner(avg(grad(u)), tensor_jump(v, n)) * dS
        - mu * inner(avg(grad(v)), tensor_jump(u, n)) * dS
    )

    a = (
        mu * inner(grad(u), grad(v)) * dx(1)
        + mu / k * inner(u, v) * dx(0)
        - div(v) * p * dx(1)
        - div(v) * p * dx(0)
        + div(u) * q * dx(0)
        + div(u) * q * dx(1)
        + stab
    )

    L = (
        inner(f, v) * dx(0)
        + inner(f, v) * dx(1)
        # - pin * dot(v, n) * ds(1)
        - pout * dot(v, n) * ds(3)
    )

    U = Function(W)

    solve(a == L, U, bcs)

    (u1, p1) = U.split()

    u_file = File(dir2 + "/u.pvd")
    u_file << u1
    p_file = File(dir2 + "/p.pvd")
    p_file << p1

    area = assemble(Constant(1.0) * ds(1))
    Q = assemble(dot(u1, n) * ds(1))
    Q_out = assemble(dot(u1, n) * ds(3))

    pin_calc = assemble(p1 * ds(1))  # pressão média na entrada
    dp = pin_calc - _pout
    k_medium = -((Q / (area * dp))) * float(mu)

    print(Q)

    print("Brinkman monopahse ---%s  seconds--" % (time.time() - start_time))
    # plt.show()

    f = open(dir2 + "/perm_abs_equivalente" + ".txt", "w")
    string = "perm_matrix =   " + str(k_matriz2 / mili_darcy)
    f.write(string)
    f.write("\n")
    string2 = "perm_equivalente =   " + str(k_medium / mili_darcy)
    f.write(string2)
    f.write("\n")
    string3 = "perm_equivalente/perm_matrix =   " + str(k_medium / k_matriz2)
    f.write(string3)
    f.write("\n")
    string4 = "Qin =   " + str(float(Q))
    f.write(string4)
    f.write("\n")
    string5 = "Qout =   " + str(float(Q_out))
    f.write(string5)
    f.write("\n")


_folder_base = [
    "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/mono_gamma35",
    # "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/vugg_sintetico/five_spot",
]


for i in _folder_base:
    try:
        os.mkdir(i)
    except OSError as error:
        print(error)

Nx = 100
Ny = 100
mu_w = 1
mu_o = 1
perm_matriz = 100  # md
dt = 1e-8
pin = 2
pout = 1
alpha = 35
# comentarios


start_time = time.time()
BrinkmanMonoBDM(perm_matriz, pin, pout, mu_w, Nx, Ny, _folder_base[0], alpha)
print(time.time() - start_time)

# BrinkmanMonoBDM(perm_matriz, pin, pout, mu_w, Nx, Ny, _folder_base[0])
# BrinkmanIMPESGsmh(_folder_base[0], mu_w, mu_o, perm_matriz, dt)
# BrinkmanIMPES(Nx, _folder_base[0], mu_w, mu_o, perm_matriz, dt, pin, pout)
