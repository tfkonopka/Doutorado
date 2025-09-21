import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure, morphology
from scipy import ndimage
import os

def matlab_to_python_mesh_converter():
    """
    Converte imagem binária para malha de elementos finitos com refinamento adaptativo
    Versão Python do código MATLAB original - Modificada para malha não estruturada
    """
    
    # Limpar variáveis (equivalente ao clear/clc)
    plt.close('all')
    
    # Definir caminhos dos arquivos
    image_input = r'/home/tfk/Desktop/Teste_ImageJ/Imagens/vug_config_R1.91mm_N225_D3.82mm_17583E12.tif'
    geo_output = r'/home/tfk/Desktop/Teste_ImageJ/ImagensConvertidas/vug_config_R1.91mm_N225_D3.82mm_17583E12_v1_adaptive.geo'
    
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
    
    # Parâmetros de malha MODIFICADOS para refinamento adaptativo
    cl_coarse = 5.0      # Tamanho grosso (bordas e regiões afastadas)
    cl_fine = 1.0        # Tamanho fino (próximo aos contornos internos)
    cl_boundary = 5.0    # Tamanho nas bordas do domínio
    
    # Distâncias para refinamento
    dist_influence = 15.0  # Distância de influência do refinamento
    dist_threshold = 5.0   # Distância limite para refinamento máximo
    
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
    
    print(f"Gerando arquivo GMsh com refinamento adaptativo: {geo_output}")
    
    # Escrever arquivo .geo
    with open(geo_output, 'w') as fileID:
        
        # Cabeçalho e informações de tamanho da malha
        fileID.write('\n')
        fileID.write('// Mesh size description - Adaptive refinement\n')
        fileID.write('\n')
        fileID.write(f'cl_coarse = {cl_coarse:.4f};  // Coarse mesh size (far from boundaries)\n')
        fileID.write(f'cl_fine   = {cl_fine:.4f};    // Fine mesh size (near internal boundaries)\n')
        fileID.write(f'cl_boundary = {cl_boundary:.4f}; // Boundary mesh size\n')
        fileID.write(f'dist_influence = {dist_influence:.4f}; // Distance of influence\n')
        fileID.write(f'dist_threshold = {dist_threshold:.4f}; // Distance threshold for max refinement\n')
        fileID.write('\n')
        
        # Pontos do contorno com tamanho de malha para bordas
        fileID.write('// Domain boundary points\n')
        fileID.write('Point(1) = {0, 0, 0, cl_boundary};\n')
        fileID.write(f'Point(2) = {{{xi:.3f}, 0, 0, cl_boundary}};\n')
        fileID.write(f'Point(3) = {{{xi:.3f}, {yi:.3f}, 0, cl_boundary}};\n')
        fileID.write(f'Point(4) = {{0, {yi:.3f}, 0, cl_boundary}};\n')
        fileID.write('\n\n')
        
        # Linhas que conectam o contorno
        fileID.write('// Lines that connect domain boundary\n')
        fileID.write('Line(1) = {1, 2};\n')
        fileID.write('Line(2) = {3, 2};\n')
        fileID.write('Line(3) = {4, 3};\n')
        fileID.write('Line(4) = {1, 4};\n')
        fileID.write('\n')
        
        # Definir coordenadas dos segmentos
        fileID.write('// Define internal boundary coordinates\n')
        
        # Processar cada contorno (começando do índice 1, equivalente ao k=2:a do MATLAB)
        internal_curves = []
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
            
            fileID.write(f'\n// Internal boundary {k}\n')
            fileID.write(f'X{k+1} = {{{x_str}}};\n')
            fileID.write(f'Y{k+1} = {{{y_str}}};\n')
        
        fileID.write('\n')
        fileID.write('// Define spline curves for internal boundaries\n')
        fileID.write('\n')
        
        # Gerar superfícies spline
        line_loop = []
        surfaces = []
        curves = []
        i = 90
        
        for k in range(1, len(contours)):
            j = i + 1
            m = j + 1
            
            fileID.write(f'nR{k} = #X{k+1}[ ];\n')
            fileID.write('p0  =  newp;\n')
            fileID.write('p   =  p0;\n')
            fileID.write(f'For i In {{0:nR{k}-1}}\n')
            fileID.write(f'  Point(newp) = {{X{k+1}[i], Y{k+1}[i], 0, cl_fine}};\n')
            fileID.write('EndFor\n')
            fileID.write('p2  =  newp-1;\n')
            fileID.write(f'BSpline({i}) = {{p:p2,p}};\n')
            fileID.write(f'Line Loop({j}) = {{{i}}};\n')
            fileID.write(f'Plane Surface({m}) = {{{j}}};\n')
            fileID.write('\n')
            
            line_loop.append(j)
            surfaces.append(m)
            curves.append(i)
            i = m + 1
        
        # Definir todas as superfícies
        fileID.write('\n')
        fileID.write('// Define domain surface\n')
        fileID.write('Line Loop(5) = {1, -2, -3, -4};\n')
        fileID.write('\n')
        
        # Superfícies finais
        if line_loop:
            line_loop_str = ','.join(map(str, line_loop))
            surfaces_str = ','.join(map(str, surfaces))
            
            fileID.write(f'Plane Surface({m+1}) = {{5,{line_loop_str}}};\n')
            fileID.write('\n')
        
        # NOVO: Configuração de refinamento adaptativo usando campos de tamanho
        fileID.write('//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
        fileID.write('// ADAPTIVE MESH REFINEMENT CONFIGURATION\n')
        fileID.write('//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
        fileID.write('\n')
        
        # Desabilitar tamanho característico das bordas para usar campos
        fileID.write('// Disable boundary layer size control to use size fields\n')
        fileID.write('Mesh.CharacteristicLengthExtendFromBoundary = 0;\n')
        fileID.write('Mesh.CharacteristicLengthFromPoints = 0;\n')
        fileID.write('Mesh.CharacteristicLengthFromCurvature = 0;\n')
        fileID.write('\n')
        
        # Criar campo de distância para cada contorno interno
        field_id = 1
        distance_fields = []
        
        if curves:
            for idx, curve_id in enumerate(curves):
                fileID.write(f'// Distance field for internal boundary {idx+1}\n')
                fileID.write(f'Field[{field_id}] = Distance;\n')
                fileID.write(f'Field[{field_id}].EdgesList = {{{curve_id}}};\n')
                fileID.write(f'Field[{field_id}].NNodesByEdge = 200;\n')
                fileID.write('\n')
                
                distance_fields.append(field_id)
                field_id += 1
        
        # Campo de limiar (threshold) baseado na distância
        if distance_fields:
            fileID.write('// Threshold field for adaptive refinement\n')
            fileID.write(f'Field[{field_id}] = Threshold;\n')
            
            # Se há múltiplos campos de distância, usar o mínimo
            if len(distance_fields) == 1:
                fileID.write(f'Field[{field_id}].InField = {distance_fields[0]};\n')
            else:
                # Criar campo mínimo
                min_field_id = field_id + 1
                distance_fields_str = ','.join(map(str, distance_fields))
                
                fileID.write(f'// Minimum distance field\n')
                fileID.write(f'Field[{min_field_id}] = Min;\n')
                fileID.write(f'Field[{min_field_id}].FieldsList = {{{distance_fields_str}}};\n')
                fileID.write('\n')
                fileID.write(f'Field[{field_id}].InField = {min_field_id};\n')
                field_id += 1
            
            # Parâmetros do threshold
            fileID.write(f'Field[{field_id}].SizeMin = cl_fine;\n')
            fileID.write(f'Field[{field_id}].SizeMax = cl_coarse;\n')
            fileID.write(f'Field[{field_id}].DistMin = dist_threshold;\n')
            fileID.write(f'Field[{field_id}].DistMax = dist_influence;\n')
            fileID.write('\n')
            
            threshold_field = field_id
            field_id += 1
        else:
            # Caso não haja contornos internos, usar tamanho uniforme
            fileID.write('// Uniform field (no internal boundaries detected)\n')
            fileID.write(f'Field[{field_id}] = MathEval;\n')
            fileID.write(f'Field[{field_id}].F = "cl_coarse";\n')
            threshold_field = field_id
            field_id += 1
        
        # Campo para refinamento nas bordas do domínio
        fileID.write('// Boundary refinement field\n')
        fileID.write(f'Field[{field_id}] = Distance;\n')
        fileID.write(f'Field[{field_id}].EdgesList = {{1,2,3,4}};\n')
        fileID.write(f'Field[{field_id}].NNodesByEdge = 100;\n')
        fileID.write('\n')
        
        boundary_dist_field = field_id
        field_id += 1
        
        fileID.write(f'Field[{field_id}] = Threshold;\n')
        fileID.write(f'Field[{field_id}].InField = {boundary_dist_field};\n')
        fileID.write(f'Field[{field_id}].SizeMin = cl_boundary;\n')
        fileID.write(f'Field[{field_id}].SizeMax = cl_coarse;\n')
        fileID.write(f'Field[{field_id}].DistMin = 2.0;\n')
        fileID.write(f'Field[{field_id}].DistMax = 20.0;\n')
        fileID.write('\n')
        
        boundary_threshold_field = field_id
        field_id += 1
        
        # Campo final: mínimo entre refinamento interno e das bordas
        fileID.write('// Final combined field\n')
        fileID.write(f'Field[{field_id}] = Min;\n')
        fileID.write(f'Field[{field_id}].FieldsList = {{{threshold_field},{boundary_threshold_field}}};\n')
        fileID.write('\n')
        
        final_field = field_id
        
        # Aplicar o campo de tamanho
        fileID.write(f'// Apply the size field\n')
        fileID.write(f'Background Field = {final_field};\n')
        fileID.write('\n')
        
        # Configurações adicionais da malha
        fileID.write('// Additional mesh options\n')
        fileID.write('Mesh.Algorithm = 6;           // Frontal-Delaunay\n')
        fileID.write('Mesh.RecombineAll = 0;        // Keep triangular elements\n')
        fileID.write('Mesh.CharacteristicLengthMin = 1.0;\n')
        fileID.write('Mesh.CharacteristicLengthMax = 12.0;\n')
        fileID.write('Mesh.ElementOrder = 1;        // Linear elements\n')
        fileID.write('Mesh.Smoothing = 5;           // Smoothing iterations\n')
        fileID.write('\n')
        
        # Definições físicas
        fileID.write('// Physical definitions\n')
        fileID.write('Physical Line(1) = {4};  // Left boundary\n')
        fileID.write('Physical Line(2) = {3};  // Top boundary\n')
        fileID.write('Physical Line(3) = {2};  // Right boundary\n')
        fileID.write('Physical Line(4) = {1};  // Bottom boundary\n')
        fileID.write('\n')
        
        if line_loop:
            fileID.write(f'Physical Surface(0) = {{{m+1}}};      // Solid matrix\n')
            fileID.write(f'Physical Surface(1) = {{{surfaces_str}}}; // Pores/voids\n')
        else:
            fileID.write(f'Physical Surface(0) = {{{m+1}}};      // Uniform domain\n')
    
    print(f"Arquivo {geo_output} gerado com sucesso!")
    print("Características da malha:")
    print(f"  - Tamanho fino (próximo aos poros): {cl_fine}")
    print(f"  - Tamanho grosso (regiões afastadas): {cl_coarse}")
    print(f"  - Tamanho nas bordas: {cl_boundary}")
    print(f"  - Distância de influência: {dist_influence}")
    print(f"  - Limiar de refinamento: {dist_threshold}")
    
    # Abrir o arquivo (equivalente ao uiopen)
    try:
        os.startfile(geo_output)
    except:
        print(f"Arquivo salvo em: {geo_output}")

# Função principal
if __name__ == "__main__":
    matlab_to_python_mesh_converter()