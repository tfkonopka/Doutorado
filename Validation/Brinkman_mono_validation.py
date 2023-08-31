from fenics import *
import time
import ufl

# from brinkman_biphase_IMPES_v2_lib import *


class Obstacle(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (0.08, 0.1)) and between(x[0], (0.0, 0.025))


def tensor_jump(v, n):
    return ufl.outer(v, n)("+") + ufl.outer(v, n)("-")


def BrinkmanMonoBDM(k_matriz, pin, pout, mu, Nx, Ny, caminho):
    start_time = time.time()

    dir2 = caminho + "/monophase"
    num_refinements = 3
    mesh = RectangleMesh(Point(0, 0), Point(1, 1), Nx, Ny, "right")
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

    obstacle = Obstacle()
    domains = MeshFunction("size_t", mesh, mesh.topology().dim())
    domains.set_all(0)
    obstacle.mark(domains, 1)

    # bc1 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 1)
    bc2 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 2)
    # # # bc3 = DirichletBC(VQ.sub(0), Constant((0.0, 0.0)), boundaries, 3)
    bc4 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 4)

    bcs = [bc2, bc4]  # velocity BC

    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    dx = Measure("dx", domain=mesh, subdomain_data=domains)

    mili_darcy = 9.86923e-16  # Porous media permeability convertion factor md [m2] (1 Darcy = E-12 m2)

    k_matriz2 = k_matriz
    kgf_cm2_to_Pa = 98066.5

    pin = pin * kgf_cm2_to_Pa  # Pa
    pout = pout * kgf_cm2_to_Pa  # Pa

    mu = Constant(mu)
    k = Constant(k_matriz2)
    pin = Constant(pin)
    pout = Constant(pout)
    dp = pin - pout
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

    nitche = (
        alpha / h * inner(outer(v, n), outer(u, n)) * ds(2)
        - inner(grad(u), outer(v, n)) * ds(2)
        - inner(grad(v), outer(u, n)) * ds(2)
        # + alpha / h * inner(outer(v, n), outer(u, n)) * ds(4)
        # - inner(grad(u), outer(v, n)) * ds(4)
        # - inner(grad(v), outer(u, n)) * ds(4)
    )

    a = (
        mu * inner(grad(u), grad(v)) * dx(1)
        + mu / k * inner(u, v) * dx(0)
        - div(v) * p * dx(1)
        - div(v) * p * dx(0)
        + div(u) * q * dx(0)
        + div(u) * q * dx(1)
        + stab
        + nitche
    )

    L = (
        inner(f, v) * dx(0)
        + inner(f, v) * dx(1)
        - pin * dot(v, n) * ds(1)
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

    dp = float(pin - pout)
    k_medium = -((Q / (area * dp))) * float(mu)

    print(Q)

    print("Brinkman monopahse ---%s  seconds--" % (time.time() - start_time))
    # plt.show()

    f = open(dir2 + "/perm_abs_equivalente" + ".txt", "w")
    string = "perm_matrix =   " + str(k_matriz)
    f.write(string)
    f.write("\n")
    string2 = "perm_equivalente =   " + str(k_medium)
    f.write(string2)
    f.write("\n")
    string3 = "perm_equivalente/perm_matrix =   " + str(k_medium / k_matriz)
    f.write(string3)
    f.write("\n")
