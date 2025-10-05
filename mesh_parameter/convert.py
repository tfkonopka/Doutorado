import os
from dolfin import *
import meshio

file_in = "/home/tfk/Desktop/Teste_ImageJ/ImagensConvertidas/vug_config_R1.91mm_N225_D3.82mm_AFA48F"
file_out = "/home/tfk/Desktop/Teste_ImageJ/ImagensConvertidas/teste_1/vug_config_R1.91mm_N225_D3.82mm_AFA48F_v2"

# Criar diretório de saída se não existir
os.makedirs(os.path.dirname(file_out), exist_ok=True)

# Gerar mesh com GMSH
os.system("gmsh " + file_in + ".geo -2 -o " + file_out + ".msh")

# Converter com meshio (corrigindo os espaços)
os.system("meshio-convert " + file_out + ".msh " + file_out + ".xdmf")

# Ler o arquivo MSH
msh = meshio.read(file_out + ".msh")

# Verificar se as células existem no arquivo
if "triangle" not in msh.cells_dict:
    raise ValueError("Arquivo MSH não contém células triangulares")

# Escrever arquivos XDMF
meshio.write(
    file_out + "_mesh.xdmf",
    meshio.Mesh(
        points=msh.points[:, :2], 
        cells=[("triangle", msh.cells_dict["triangle"])]
    ),
)

# Escrever boundaries (verificar se existem linhas)
if "line" in msh.cells_dict:
    meshio.write(
        file_out + "_boundaries.xdmf",
        meshio.Mesh(
            points=msh.points[:, :2],
            cells=[("line", msh.cells_dict["line"])],
            cell_data={"bnd_marker": [msh.cell_data_dict["gmsh:physical"]["line"]]},
        ),
    )

# Escrever domains
meshio.write(
    file_out + "_domains.xdmf",
    meshio.Mesh(
        points=msh.points[:, :2],
        cells=[("triangle", msh.cells_dict["triangle"])],
        cell_data={"dom_marker": [msh.cell_data_dict["gmsh:physical"]["triangle"]]},
    ),
)

print("Conversão concluída com sucesso!")