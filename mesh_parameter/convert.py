import os
from dolfin import *
import meshio


file_in = "am8_c2_s-1240"
file_out = "holesuCT"

os.system("gmsh " + file_in + ".geo -2 -o " + file_out + ".msh")
os.system("meshio-convert" + "holesuCT.msh" + file_out + ".xdmf")

msh = meshio.read("holesuCT.msh")
tri_data = msh.cell_data_dict["gmsh:physical"]["triangle"]
line_data = msh.cell_data_dict["gmsh:physical"]["line"]

meshio.write(
    "mesh.xdmf",
    meshio.Mesh(
        points=msh.points[:, :2], cells={"triangle": msh.cells_dict["triangle"]}
    ),
)

meshio.write(
    "boundaries.xdmf",
    meshio.Mesh(
        points=msh.points,
        cells={"line": msh.cells_dict["line"]},
        cell_data={"bnd_marker": [msh.cell_data_dict["gmsh:physical"]["line"]]},
    ),
)

meshio.write(
    "domains.xdmf",
    meshio.Mesh(
        points=msh.points,
        cells={"triangle": msh.cells_dict["triangle"]},
        cell_data={"dom_marker": [msh.get_cell_data("gmsh:physical", "triangle")]},
    ),
)
