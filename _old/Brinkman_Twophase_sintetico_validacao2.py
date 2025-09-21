"Versão com multiplos vugg sintéticos. dt fixo em 200s. "


from fenics import *
import time
import ufl_legacy as ufl


import os


def DataRecord(t, dt, Qo, Qw, pin, pout, Vinj, Sin, Sout, Sdx, dir1):
    f = open(dir1 + "/results_2" + ".txt", "w")
    string = (
        "time"
        + ","
        + "dt"
        + ","
        + "Qo"
        + ","
        + "Qw"
        + ","
        + "pin"
        + ","
        + "pout"
        + ","
        + "Vinj"
        + ","
        + "Sin"
        + ","
        + "Sout"
        + ","
        + "Sdx"
    )
    f.write(string)
    f.write("\n")
    for i in range(len(t)):
        string = (
            str(t[i])
            + ","
            + str(float(dt[i]))
            + ","
            + str(Qo[i])
            + ","
            + str(Qw[i])
            + ","
            + str(pin[i])
            + ","
            + str(pout)
            + ","
            + str(Vinj[i])
            + ","
            + str(Sin[i])
            + ","
            + str(Sout[i])
            + ","
            + str(Sdx[i])
        )
        f.write(string)
        f.write("\n")
    f.close()


def DataRecord2(t, dt, Qo, Qw, pin, pout, Vinj, Sin, Sout, Sdx, dir1):
    f = open(dir1 + "/results_" + ".txt", "a")

    string = (
        str(t)
        + ","
        + str(dt)
        + ","
        + str(Qo)
        + ","
        + str(Qw)
        + ","
        + str(pin)
        + ","
        + str(pout)
        + ","
        + str(Vinj)
        + ","
        + str(Sin)
        + ","
        + str(Sout)
        + ","
        + str(Sdx)
    )
    f.write(string)
    f.write("\n")
    f.close()


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


def lmbdainv(s, mu_w, mu_o, no, nw):
    return 1.0 / ((s**nw) / mu_w + ((1.0 - s) ** no) / mu_o)


# Fractional flow function
def F(s, mu_rel, no, nw):
    return s**nw / (s**nw + mu_rel * (1.0 - s) ** no)


def F_vugg(s):
    return s


def mu_brinkman(s, mu_o, mu_w):
    return s * mu_w + (1.0 - s) * mu_o


class Obstacle(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[0], (0.4, 0.6)) and between(x[1], (0.1, 0.9))


# class Obstacle1(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.2, 0.430940108)) and between(x[0], (0.6, 0.830940108))


# class Obstacle2(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.6, 0.830940108)) and between(x[0], (0.6, 0.830940108))


# class Obstacle3(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.6, 0.8)) and between(x[0], (0.2, 0.4))


# class Obstacle(SubDomain):
#     def inside(self, x, on_boundary):
#         return between(x[1], (0.3, 0.7)) and between(x[0], (0.3, 0.7))

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
    mu_b = 0.0000001  # Pa.s

    mili_darcy = 9.86923e-16  # Porous media permeability convertion factor md [m2] (1 Darcy = E-12 m2)

    k_matriz = perm_darcy * mili_darcy

    kgf_cm2_to_Pa = 98066.5
    pin = pin * kgf_cm2_to_Pa  # Pa
    pout = pout * kgf_cm2_to_Pa  # Pa

    dt = Constant(dt)  # s

    phi = 1
    mu_rel = mu_w / mu_o

    sbar = Constant(1)

    Kinv = Constant(1 / 1)
    mu_b = 0.000001
    mu = Constant(mu)
    mu_b = Constant(mu_b)
    t = 0

    mesh = UnitSquareMesh(Nx, Ny, "crossed")
    order = 1
    V = FiniteElement("BDM", mesh.ufl_cell(), order)
    Q = FiniteElement("DG", mesh.ufl_cell(), order - 1)
    R1 = FiniteElement("DG", mesh.ufl_cell(), order - 1)

    Element = V * Q
    W = FunctionSpace(mesh, Element)
    R = FunctionSpace(mesh, R1)

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    s = TrialFunction(R)
    r = TestFunction(R)

    U = Function(W)
    (u_, p_) = U.split()
    S = Function(R)
    s0 = Function(R)

    s0.vector()[:] = 0.0

    # ============= DEFINITION OF SPATIALLY-VARYING PARAMETERS =====================
    marker_inner = 1
    marker_outer = 0

    no_outer = 2
    nw_outer = 2
    no_inner = 2
    nw_inner = 2

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
    domains.set_all(marker_outer)
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

    no = {marker_inner: no_inner, marker_outer: no_outer}
    nw = {marker_inner: nw_inner, marker_outer: nw_outer}

    VVV = FunctionSpace(mesh, "DG", 0)

    noo = PiecewiseConstant(no, domains)
    noo_proj = project(noo, VVV)
    nww = PiecewiseConstant(nw, domains)
    nww_proj = project(nww, VVV)

    # =========== END DEFINITION OF SPATIALLY-VARYING PARAMETERS ===================

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

    bc1 = DirichletBC(W.sub(0), Constant((1.0e-6, 0.0)), boundaries, 1)
    bc2 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 2)
    bc3 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 3)
    bc4 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 4)

    bcs = [bc1, bc3, bc2, bc4]  # velocity BC

    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    dx = Measure("dx", domain=mesh, subdomain_data=domains)

    File(dir1 + "/domains.pvd") << domains

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

    a = (
        mu_b * inner(grad(u / F(s0, mu_rel, noo_proj, nww_proj)), grad(v)) * dx(1)
        + inner(v, Kinv * u / F(s0, mu_rel, noo_proj, nww_proj)) * dx(0)
        + mu_b * inner(grad(u / F(s0, mu_rel, noo_proj, nww_proj)), grad(v)) * dx(0)
        + inner(v, Kinv * u / F(s0, mu_rel, noo_proj, nww_proj)) * dx(1)
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
        # - pout * dot(v, n) * ds(3)
    )

    un = 0.5 * (inner(u_, n) + abs(inner(u_, n)))
    un_h = 0.5 * (inner(u_, n) - abs(inner(u_, n)))

    stabilisation = (
        dt("+") * inner(jump(r), jump(un * F(s0, mu_rel, noo_proj, nww_proj))) * dS
    )

    L3 = (
        r * (s - s0) * dx(0)
        + r * (s - s0) * dx(1)
        - dt * inner(grad(r), u_) * dx(0)
        - dt * inner(grad(r), u_) * dx(1)
        + dt * r * un * ds
        + stabilisation
        + dt * r * un_h * sbar * ds(1)
    )

    a_s, L_f = lhs(L3), rhs(L3)

    u_file = XDMFFile(dir2 + "/velocity.xdmf")
    p_file = XDMFFile(dir2 + "/pressure.xdmf")
    s_file = XDMFFile(dir2 + "/saturation.xdmf")

    step = 0

    # while t < T:
    while step < 2e2:
        # ===
        _start_time = time.time()

        t += float(dt)
        solve(a == L, U, bcs)
        solve(a_s == L_f, S)
        s0.assign(S)

        if step % 2 == 0:
            p_file.write(p_, t)
            s_file.write(S, t)
            u_file.write(u_, t)

        step = step + 1


_folder_base = [
    "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/validacaoCocliteHperm2_Mu1e10",
    # "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/vugg_sintetico/five_spot",
]


for i in _folder_base:
    try:
        os.mkdir(i)
    except OSError as error:
        print(error)

Nx = 10
Ny = 10
mu_w = 1
mu_o = 1
perm_matriz = 1  # md
dt = 400
pin = 2
pout = 1
# comentarios


start_time = time.time()
# BrinkmanIMPES(_folder_base[0], mu_w, mu_o, perm_matriz, dt)
BrinkmanIMPES(Nx, _folder_base[0], mu_w, mu_o, perm_matriz, dt, pin, pout)
print(time.time() - start_time)
