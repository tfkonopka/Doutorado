from fenics import *
import time
import ufl
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
    return 1.0 / ((s ** nw) / mu_w + ((1.0 - s) ** no) / mu_o)


# Fractional flow function
def F(s, mu_rel, no, nw):
    return s ** nw / (s ** nw + mu_rel * (1.0 - s) ** no)


def F_vugg(s):
    return s


def mu_brinkman(s, mu_o, mu_w):
    return s * mu_w + (1.0 - s) * mu_o


def BrinkmanIMPESGsmh(_folder_base, mu_w, mu_o, perm_darcy, dt):

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
    pin = 2 * kgf_cm2_to_Pa  # Pa
    pout = kgf_cm2_to_Pa  # Pa

    dt = Constant(dt)  # s

    phi = 0.2
    mu_rel = mu_w / mu_o

    sbar = Constant(1)

    Kinv = Constant(1 / k_matriz)

    mu = Constant(mu)
    mu_b = Constant(mu_b)
    t = 0
    T = 5 * float(dt)

    mesh = Mesh()
    with XDMFFile(_folder_base + "/mesh/mesh.xdmf") as infile:
        infile.read(mesh)

    mvc = MeshValueCollection("size_t", mesh, 2)
    with XDMFFile(_folder_base + "/mesh/domains.xdmf") as infile:
        infile.read(mvc)
    Markers = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    mvc = MeshValueCollection("size_t", mesh, 1)
    with XDMFFile(_folder_base + "/mesh/boundaries.xdmf") as infile:
        infile.read(mvc)
    boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)

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
    no_inner = 1
    nw_inner = 1

    no = {marker_inner: no_inner, marker_outer: no_outer}
    nw = {marker_inner: nw_inner, marker_outer: nw_outer}

    VVV = FunctionSpace(mesh, "DG", 0)

    noo = PiecewiseConstant(no, Markers)
    noo_proj = project(noo, VVV)
    nww = PiecewiseConstant(nw, Markers)
    nww_proj = project(nww, VVV)

    # =========== END DEFINITION OF SPATIALLY-VARYING PARAMETERS ===================

    bc1 = DirichletBC(W.sub(0), Constant((1.0e-6, 0.0)), boundaries, 1)
    bc2 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 2)
    # # # bc3 = DirichletBC(VQ.sub(0), Constant((0.0, 0.0)), boundaries, 3)
    bc4 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 4)

    bcs = [bc1, bc2, bc4]  # velocity BC

    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    dx = Measure("dx", domain=mesh, subdomain_data=Markers)

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
        mu_brinkman(s0, mu_o=mu_o, mu_w=mu_w) * inner(grad(u), grad(v)) * dx(1)
        + inner(v, lmbdainv(s0, mu_w, mu_o, no_outer, nw_outer) * Kinv * u) * dx(0)
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

    un = 0.5 * (inner(u_, n) + abs(inner(u_, n)))
    un_h = 0.5 * (inner(u_, n) - abs(inner(u_, n)))

    stabilisation = (
        dt("+") * inner(jump(r), jump(un * F(s0, mu_rel, noo_proj, nww_proj))) * dS
    )

    L3 = (
        phi * r * (s - s0) * dx(0)
        + r * (s - s0) * dx(1)
        - dt * inner(grad(r), F(s0, mu_rel, noo_proj, nww_proj) * u_) * dx(0)
        - dt * inner(grad(r), F_vugg(s0) * u_) * dx(1)
        + dt * r * F(s0, mu_rel, no_outer, nw_outer) * un * ds
        + stabilisation
        + dt * r * un_h * sbar * ds(1)
    )

    a_s, L_f = lhs(L3), rhs(L3)

    u_file = XDMFFile(dir2 + "/velocity.xdmf")
    p_file = XDMFFile(dir2 + "/pressure.xdmf")
    s_file = XDMFFile(dir2 + "/saturation.xdmf")

    s_txt = []

    Q_dot_vector = []
    S_mean_dx_vector = []
    S_mean_in_vector = []
    S_mean_out_vector = []
    Qinj = 0
    Nw_inj = []
    Qdotw_vector = []
    Qdoto_vector = []
    vector_step = []
    t_cumulative = []

    step = 0
    parada = 1
    pin_vector = []
    dt_vector = []

    A_in = float(assemble(1 * ds(1)))
    Len = float(assemble(1 * ds(2)))
    Area = float(assemble(1 * dx))

    if os.path.exists(dir1 + "/results_" + ".txt"):
        os.remove(dir1 + "/results_" + ".txt")
    else:
        print("The file does not exist")

    DataRecord2(
        "time", "dt", "Qo", "Qw", "pin", "pout", "Vinj", "Sin", "Sout", "Sdx", dir1
    )

    # while t < T:
    while step < 1e6:
        # ===
        t += float(dt)
        solve(a == L, U, bcs)
        solve(a_s == L_f, S)
        s0.assign(S)

        if step % 100 == 0:

            p_file.write(p_, t)
            s_file.write(S, t)
            u_file.write(u_, t)

        t_cumulative.append(float(t))
        dt_vector.append(dt)

        S_mean_in_vector.append(
            assemble(S * ds(1)) / A_in
        )  # saturação média na entrada do meio poroso em t
        S_mean_out_vector.append(
            assemble(S * ds(3)) / A_in
        )  # saturação média na saida do meio poroso em t
        S_mean_dx_vector.append(
            assemble(S * dx) / Area
        )  # saturação média do meio poroso em t
        Q_dot_vector.append(
            float(-assemble(dot(u_, n) * ds(1)))
        )  # vetor com a vazão instante por passo de t

        Qinj = Qinj + Q_dot_vector[step] * float(
            dt
        )  # vazão acumulada total de injeção final
        Nw_inj.append(Qinj)

        Qdotw_vector.append(
            assemble(F(s0, mu_rel, noo_proj, nww_proj) * dot(u_, n) * ds(3))
        )  # vazão de água na saída do meio poroso

        print(f"Qdotw_vector[step] = {Qdotw_vector[step]}")

        Qdoto_vector.append(
            Q_dot_vector[step] - Qdotw_vector[step]
        )  # vazão de óleo na saída do meio poroso

        pin = assemble(p_ * ds(1)) / A_in  # pressão média na entrada
        pin_vector.append(pin)  # vetor de pressão média na entrada por

        vector_step.append(step)

        if S_mean_out_vector[step] > 0.3:
            parada = Qdoto_vector[step] / Qdotw_vector[step]
            if parada < 0.05:
                break
        else:
            parada = 1

        print(f"S_mean_dx_vector = {S_mean_dx_vector[step]}")
        print(f"S_mean_in_vector[{step}]  = {S_mean_in_vector[step]}  ")
        print(f"S_mean_out_vector[{step}] = {S_mean_out_vector[step]} ")

        print(f"step = {step}")
        print(f"parada = {parada}")

        uin = assemble(dot(u_, n) * ds(1))
        uout = assemble(dot(u_, n) * ds(3))
        erro_mat_bal = abs(abs(uin) - abs(uout))
        print(f"uin = {uin} ; out = {uout} ; erro_mat_bal = {erro_mat_bal}")

        DataRecord2(
            t_cumulative[step],
            float(dt_vector[step]),
            Qdoto_vector[step],
            Qdotw_vector[step],
            pin_vector[step],
            pout,
            Nw_inj[step],
            S_mean_in_vector[step],
            S_mean_out_vector[step],
            S_mean_dx_vector[step],
            dir1,
        )

        step = step + 1

    print(f"pressao in = {pin}")
    print(f"delta pressao = {pin - pout}")
    DataRecord(
        t_cumulative,
        dt_vector,
        Qdoto_vector,
        Qdotw_vector,
        pin_vector,
        pout,
        Nw_inj,
        S_mean_in_vector,
        S_mean_out_vector,
        S_mean_dx_vector,
        dir1,
    )
