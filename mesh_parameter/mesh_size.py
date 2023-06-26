from fenics import *


_folder_base = (
    "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/Arapua/teste_malha/Arapua17"
)


def mesh_size(caminho):

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

    h_max = mesh.hmax()
    h_min = mesh.hmin()
    n_ele = mesh.num_cells()

    print(h_max, h_min, n_ele)


mesh_size(_folder_base)
