"Versão com multiplos vugg sintéticos. dt fixo em 200s. "


from fenics import *
import time
import ufl
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


class Obstacle(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (0.3, 0.7)) and between(x[0], (0.3, 0.7))


def tensor_jump(v, n):
    return ufl.outer(v, n)("+") + ufl.outer(v, n)("-")


def lmbdainv(s, mu_w, mu_o, no, nw):
    return 1.0 / ((s**nw) / mu_w + ((1.0 - s) ** no) / mu_o)


# Fractional flow function
def F(s, mu_rel, no, nw):
    return s**nw / (s**nw + mu_rel * (1.0 - s) ** no)


def DarcyIMPESRT(Nx, _folder_base, mu_w, mu_o, perm_darcy, perm_vugg, dt):

    Ny = Nx
    dir0 = _folder_base + "/Darcy_2_domai_RT"
    dir1 = dir0 + "/dir1"
    dir2 = dir0 + "/dir2"

    try:
        os.mkdir(dir0)
    except OSError as error:
        print(error)

    try:
        os.mkdir(dir1)
    except OSError as error:
        print(error)

    try:
        os.mkdir(dir2)
    except OSError as error:
        print(error)

    mu = (mu_o + mu_w) / 2  # Pa.s

    mili_darcy = 9.86923e-16  # Porous media permeability convertion factor md [m2] (1 Darcy = E-12 m2)

    k_matriz = perm_darcy * mili_darcy
    k_vugg = perm_vugg * mili_darcy

    kgf_cm2_to_Pa = 98066.5
    pin = 2 * kgf_cm2_to_Pa  # Pa
    pout = kgf_cm2_to_Pa  # Pa

    dt = Constant(dt)  # s

    phi = 0.2
    mu_rel = mu_w / mu_o

    sbar = Constant(1)

    Kinv_matriz = Constant(1 / k_matriz)
    Kinv_vugg = Constant(1 / k_vugg)

    mu = Constant(mu)

    t = 0
    T = 5 * float(dt)

    mesh = mesh = UnitSquareMesh(Nx, Ny, "crossed")

    order = 1
    V = FiniteElement("RT", mesh.ufl_cell(), order)
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
    no_inner = 1
    nw_inner = 1

    obstacle = Obstacle()

    Markers = MeshFunction("size_t", mesh, mesh.topology().dim())
    Markers.set_all(marker_outer)
    obstacle.mark(Markers, marker_inner)

    no = {marker_inner: no_inner, marker_outer: no_outer}
    nw = {marker_inner: nw_inner, marker_outer: nw_outer}

    VVV = FunctionSpace(mesh, "DG", 0)

    noo = PiecewiseConstant(no, Markers)
    noo_proj = project(noo, VVV)
    nww = PiecewiseConstant(nw, Markers)
    nww_proj = project(nww, VVV)

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

    # =========== END DEFINITION OF SPATIALLY-VARYING PARAMETERS ===================

    bc1 = DirichletBC(W.sub(0), Constant((1.0e-6, 0.0)), boundaries, 1)
    bc2 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 2)
    # # # bc3 = DirichletBC(VQ.sub(0), Constant((0.0, 0.0)), boundaries, 3)
    bc4 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 4)

    bcs = [bc1, bc2, bc4]  # velocity BC

    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    dx = Measure("dx", domain=mesh, subdomain_data=Markers)

    File(dir1 + "/domains.pvd") << Markers

    alpha = 35
    h = CellDiameter(mesh)

    f = Constant((0.0, 0.0))
    n = FacetNormal(mesh)

    a = (
        inner(v, lmbdainv(s0, mu_w, mu_o, no_inner, nw_inner) * Kinv_matriz * u) * dx(0)
        + inner(v, lmbdainv(s0, mu_w, mu_o, no_outer, nw_outer) * Kinv_vugg * u) * dx(1)
        - div(v) * p * dx(1)
        - div(v) * p * dx(0)
        + div(u) * q * dx(0)
        + div(u) * q * dx(1)
    )

    L = (
        inner(f, v) * dx(0)
        + inner(f, v) * dx(1)
        # - pin * dot(v, n) * ds(1)
        - pout * dot(v, n) * ds(3)
    )

    un = 0.5 * (inner(u_, n) + abs(inner(u_, n)))
    un_h = 0.5 * (inner(u_, n) - abs(inner(u_, n)))

    stabilisation = (
        dt("+") * inner(jump(r), jump(un * F(s0, mu_rel, noo_proj, nww_proj))) * dS
    )

    L3 = (
        phi * r * (s - s0) * dx(0)
        + r * (s - s0) * dx(1)
        - dt * inner(grad(r), F(s0, mu_rel, noo_proj, nww_proj) * u_) * dx
        + dt * r * F(s0, mu_rel, no_outer, nw_outer) * un * ds
        + stabilisation
        + dt * r * un_h * sbar * ds(1)
    )

    a_s, L_f = lhs(L3), rhs(L3)

    u_file = XDMFFile(dir2 + "/velocity.xdmf")
    p_file = XDMFFile(dir2 + "/pressure.xdmf")
    s_file = XDMFFile(dir2 + "/saturation.xdmf")

    step = 0

    while step < 1e5:
        # ===
        t += float(dt)
        solve(a == L, U, bcs)
        solve(a_s == L_f, S)
        s0.assign(S)
        if step % 50 == 0:
            p_file.write(p_, t)
            s_file.write(S, t)
            u_file.write(u_, t)

        step = step + 1
