import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure, morphology
from scipy import ndimage
import os

def matlab_to_python_mesh_converter():
    """
    Converte imagem binária para malha de elementos finitos
    Versão Python do código MATLAB original
    """
    
    # Limpar variáveis (equivalente ao clear/clc)
    plt.close('all')
    
    # Definir caminhos dos arquivos
    image_input = r'/home/tfk/Desktop/Teste_ImageJ/Imagens/vug_config_R1.91mm_N225_D3.82mm_17583E12.tif'
    geo_output = r'/home/tfk/Desktop/Teste_ImageJ/ImagensConvertidas/vug_config_R1.91mm_N225_D3.82mm_17583E12_v1.geo'
    
    # Ler e processar a imagem
    print("Processando imagem...")
    I = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
    
    # Converter para binário (equivalente ao im2bw)
    _, BW = cv2.threshold(I, 128, 255, cv2.THRESH_BINARY)
    BW = BW.astype(bool)
    
    # Preencher buracos (equivalente ao imfill)
    Ifill = ndimage.binary_fill_holes(BW)
    
    # Encontrar contornos (equivalente ao bwboundaries)
    contours, hierarchy = cv2.findContours(
        BW.astype(np.uint8) * 255, 
        cv2.RETR_TREE, 
        cv2.CHAIN_APPROX_NONE
    )
    
    # Obter dimensões da imagem
    NyI, NxI = Ifill.shape
    
    # Mostrar imagem binária
    plt.figure(figsize=(10, 8))
    plt.imshow(BW, cmap='gray')
    plt.title('Imagem Binária')
    plt.axis('off')
    plt.show()
    
    # Plotar contornos
    plt.figure(figsize=(10, 8))
    plt.hold = True  # Equivalente ao hold on
    
    # Processar contornos e hierarquia
    u = []
    v = []
    
    if hierarchy is not None:
        hierarchy = hierarchy[0]  # OpenCV retorna array 3D
        
        for k, contour in enumerate(contours):
            # Verificar se é contorno pai (tem filhos)
            if hierarchy[k][2] != -1:  # Tem filhos
                u.append(k)
                # Plotar contorno pai em vermelho
                contour_points = contour.reshape(-1, 2)
                y_coords = NyI - contour_points[:, 1]  # Inverter Y
                plt.plot(contour_points[:, 0], y_coords, 'r-', linewidth=2)
                
                # Encontrar e plotar filhos em verde
                child_idx = hierarchy[k][2]
                while child_idx != -1:
                    v.append(child_idx)
                    child_contour = contours[child_idx].reshape(-1, 2)
                    y_child = NyI - child_contour[:, 1]
                    plt.plot(child_contour[:, 0], y_child, 'g-', linewidth=2)
                    child_idx = hierarchy[child_idx][0]  # próximo irmão
    
    plt.title('Contornos Detectados (Vermelho: Externos, Verde: Internos)')
    plt.axis('equal')
    plt.show()
    
    # Parâmetros de malha
    cl_1 = 5
    cl_2 = 1
    
    # Escala uCT (m)
    # 2000 x 1000 pixels
    xi = 200
    yi = 100
    sc = 0.1  # escala
    
    # Extensão do domínio
    xmin = 1
    xmax = NxI
    ymin = 1
    ymax = NyI
    
    print(f"Gerando arquivo GMsh: {geo_output}")
    
    # Escrever arquivo .geo
    with open(geo_output, 'w') as fileID:
        
        # Cabeçalho e informações de tamanho da malha
        fileID.write('\n')
        fileID.write('// mesh size description\n')
        fileID.write('\n')
        fileID.write(f'cl_1   =  {cl_1:.4f};\n')
        fileID.write(f'cl_2   =  {cl_2:.4f};\n')
        fileID.write('\n')
        
        # Pontos do contorno
        fileID.write('// boundary points\n')
        fileID.write('Point(1) = {0, 0, 0, cl_1};\n')
        fileID.write(f'Point(2) = {{{xi:.3f}, 0, 0, cl_1}};\n')
        fileID.write(f'Point(3) = {{{xi:.3f}, {yi:.3f}, 0, cl_1}};\n')
        fileID.write(f'Point(4) = {{0, {yi:.3f}, 0, cl_1}};\n')
        fileID.write('\n\n')
        
        # Linhas que conectam o contorno
        fileID.write('// lines that connect boundary\n')
        fileID.write('Line(1) = {1, 2};\n')
        fileID.write('Line(2) = {3, 2};\n')
        fileID.write('Line(3) = {4, 3};\n')
        fileID.write('Line(4) = {1, 4};\n')
        fileID.write('\n')
        
        # Parâmetros da malha
        fileID.write('// Mesh Parameters\n')
        fileID.write('Mesh.CharacteristicLengthExtendFromBoundary = 0;\n')
        fileID.write('Mesh.CharacteristicLengthMax = 1.0;\n')
        fileID.write('\n')
        
        # Definir coordenadas dos segmentos
        fileID.write('// Define Segment coordinates\n')
        
        # Processar cada contorno (começando do índice 1, equivalente ao k=2:a do MATLAB)
        for k in range(1, len(contours)):
            boundary = contours[k].reshape(-1, 2)
            c1 = boundary[:, 0]  # coordenadas X
            c2 = boundary[:, 1]  # coordenadas Y
            
            # Converter para coordenadas métricas
            x = (c1) * sc
            y = (ymax - c2) * sc
            
            # Criar strings das coordenadas
            x_str = ','.join([f'{val:.6f}' for val in x])
            y_str = ','.join([f'{val:.6f}' for val in y])
            
            fileID.write(f'\n// Hole {k+1}\n')
            fileID.write(f'X{k+1} = {{{x_str}}};\n')
            fileID.write(f'Y{k+1} = {{{y_str}}};\n')
        
        fileID.write('\n')
        fileID.write('// Define spline surface\n')
        fileID.write('\n')
        fileID.write('LN = 90;\n')
        fileID.write('\n')
        
        # Gerar superfícies spline
        line_loop = []
        surfaces = []
        i = 90
        
        for k in range(1, len(contours)):
            j = i + 1
            m = j + 1
            
            fileID.write(f'nR = #X{k+1}[ ];\n')
            fileID.write('p0  =  newp;\n')
            fileID.write('p   =  p0;\n')
            fileID.write('For i In {0:nR-1}\n')
            fileID.write(f'Point(newp)  =    {{X{k+1}[i], Y{k+1}[i], 0, cl_2}};\n')
            fileID.write('EndFor\n')
            fileID.write('p2  =  newp-1;\n')
            fileID.write(f'BSpline({i})   =  {{p:p2,p}};\n')
            fileID.write(f'Line Loop({j}) = {{{i}}};\n')
            fileID.write(f'Plane Surface({m}) = {{{j}}};\n')
            fileID.write('\n')
            
            line_loop.append(j)
            surfaces.append(m)
            i = m + 1
        
        # Definir todas as superfícies
        fileID.write('\n')
        fileID.write('// Define all surfaces\n')
        fileID.write('Line Loop(5) = {1, -2, -3, -4};\n')
        fileID.write('\n')
        
        fileID.write('Physical Line(1) = {4};\n')
        fileID.write('Physical Line(2) = {3};\n')
        fileID.write('Physical Line(3) = {2};\n')
        fileID.write('Physical Line(4) = {1};\n')
        fileID.write('\n')
        
        # Superfícies finais
        if line_loop:
            line_loop_str = ','.join(map(str, line_loop))
            surfaces_str = ','.join(map(str, surfaces))
            
            fileID.write(f'Plane Surface({m+1}) = {{5,{line_loop_str}}};\n')
            fileID.write(f'Physical Surface(0) = {{{m+1}}};\n')
            fileID.write(f'Physical Surface(1) = {{{surfaces_str}}};\n')
    
    print(f"Arquivo {geo_output} gerado com sucesso!")
    
    # Abrir o arquivo (equivalente ao uiopen)
    try:
        os.startfile(geo_output)
    except:
        print(f"Arquivo salvo em: {geo_output}")

# Função principal
if __name__ == "__main__":
    matlab_to_python_mesh_converter()
