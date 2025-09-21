from fenics import *
import time

# import ufl
import ufl_legacy as ufl


def tensor_jump(v, n):
    return ufl.outer(v, n)("+") + ufl.outer(v, n)("-")


def BrinkmanMonoBDM(k_matriz, pin, pout, mu, Nx, Ny, caminho):
    start_time = time.time()

    dir2 = caminho + "/monophase"
    # dir3 = caminho + "/mesh"

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
    bc2 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 1)  # face Norte
    # bc3 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 3)  # face leste2.2.6
    bc4 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 3)  # face sul

    bcs = [bc2, bc4]  # velocity BC

    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    dx = Measure("dx", domain=mesh, subdomain_data=Markers)

    mili_darcy = 9.86923e-16  # Porous media permeability convertion factor md [m2] (1 Darcy = E-12 m2)

    k_matriz2 = k_matriz * mili_darcy

    kgf_cm2_to_Pa = 98066.5

    # pin = 2 * kgf_cm2_to_Pa  # Pa
    # pout = kgf_cm2_to_Pa  # Pa

    pin = 200000
    pout = 100000

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

    a = (
        mu * inner(grad(u), grad(v)) * dx(1)
        + mu / k * inner(u, v) * dx(0)
        # + mu / k * inner(u, v) * dx(1)
        - div(v) * p * dx(1)
        - div(v) * p * dx(0)
        + div(u) * q * dx(0)
        + div(u) * q * dx(1)
        + stab
    )

    L = (
        inner(f, v) * dx(0)
        + inner(f, v) * dx(1)
        - pin * dot(v, n) * ds(2)
        - pout * dot(v, n) * ds(4)
    )

    U = Function(W)

    solve(a == L, U, bcs)

    (u1, p1) = U.split()

    u_file = File(dir2 + "/u.pvd")
    u_file << u1
    p_file = File(dir2 + "/p.pvd")
    p_file << p1

    area = assemble(Constant(1.0) * ds(2))
    Q = assemble(dot(u1, n) * ds(2))
    print(f"Q = {float(Q)}")
    pin_calc = assemble(p1 * ds(2)) / area
    pout_calc = assemble(p1 * ds(4)) / area
    _len = assemble(Constant(1.0) * ds(1))

    dp = float(pin_calc - pout_calc) / float(_len)
    deltap = float(pin - pout)
    k_medium = -((Q / (area * dp))) * float(mu)

    print(f"Q = {Q}")
    print(f"dp/Len = {dp}")
    print(f"Len = {_len}")
    print(f"area = {area}")
    print(f"deltap = {deltap}")

    print("Brinkman monopahse ---%s  seconds--" % (time.time() - start_time))
    # plt.show()
    print(f"area inlet= {area}")

    vol_total = assemble(1 * dx)
    vol_matrix = assemble(1 * dx(0))
    vol_vugg = assemble(1 * dx(1))

    vol_total = vol_matrix + vol_vugg
    vol_phi = vol_vugg + 0.2 * vol_matrix

    print(f"vol_total = {vol_total}")
    print(f"vol_matrix = {vol_matrix}")
    print(f"vol_vugg = {vol_vugg}")
    print(f"vol_phi = {vol_phi}")
    f = open(dir2 + "/perm_abs_equivalente" + ".txt", "w")

    print(k_matriz)
    string = "perm_matrix =   " + str(k_matriz)
    f.write(string)
    f.write("\n")
    string2 = "perm_equivalente =   " + str(k_medium / mili_darcy)
    f.write(string2)
    f.write("\n")
    string3 = "perm_equivalente/perm_matrix =   " + str(
        k_medium / mili_darcy / k_matriz
    )
    f.write(string3)
    f.write("\n")
    string4 = " volume total=   " + str(vol_total)
    f.write(string4)
    f.write("\n")
    string5 = " volume matrix total =   " + str(vol_matrix)
    f.write(string5)
    f.write("\n")
    string6 = " volume vugg =   " + str(vol_vugg)
    f.write(string6)
    f.write("\n")
    string7 = " volume poroso total =   " + str(vol_phi)
    f.write(string7)
    f.write("\n")
    string8 = " porosidade total =   " + str(vol_phi / vol_total)
    f.write(string8)
    f.write("\n")

    return pin_calc, pout_calc, -Q


# (k_matriz, pin, pout, mu, Nx, Ny, caminho)


# _folder_base = "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/vugg_sintetico/vugg_sens/Araupa"

# BrinkmanMonoBDM(1e-6, 10, 1, 1, 1, 1, _folder_base)
