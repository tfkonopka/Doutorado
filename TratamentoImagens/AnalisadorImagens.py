import cv2
import numpy as np
from scipy import ndimage
import csv

def analisar_objetos_pretos(caminho_imagem):
    """
    Analisa uma imagem para identificar objetos pretos em fundo branco.
    
    Args:
        caminho_imagem: Caminho para o arquivo de imagem
    
    Returns:
        tuple: (numero_objetos, lista_pixels_por_objeto, rotulos, imagem_original)
    """
    # Carregar a imagem
    imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
    
    if imagem is None:
        raise ValueError(f"Não foi possível carregar a imagem: {caminho_imagem}")
    
    # Binarizar a imagem (pixels pretos = 1, brancos = 0)
    _, imagem_binaria = cv2.threshold(imagem, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Identificar objetos conectados
    rotulos, num_objetos = ndimage.label(imagem_binaria)
    
    # Contar pixels de cada objeto
    pixels_por_objeto = []
    for i in range(1, num_objetos + 1):
        num_pixels = np.sum(rotulos == i)
        pixels_por_objeto.append(num_pixels)
    
    return num_objetos, pixels_por_objeto, rotulos, imagem


def ajustar_elipse_minima(pontos):
    """
    Ajusta uma elipse que contenha todos os pontos dados.
    Usa cv2.fitEllipse e depois expande a elipse para garantir que todos os pontos estejam dentro.
    
    Args:
        pontos: Array de pontos (N, 2) com coordenadas dos pixels
    
    Returns:
        tuple: Parâmetros da elipse (centro, eixos, angulo) ou None
    """
    if len(pontos) < 5:
        return None
    
    # Primeiro, ajusta uma elipse inicial
    elipse = cv2.fitEllipse(pontos)
    centro, eixos, angulo = elipse
    
    # Converter ângulo para radianos
    angulo_rad = np.deg2rad(angulo)
    
    # Calcular a distância máxima de qualquer ponto ao centro da elipse
    # na direção normalizada pelos eixos da elipse
    cx, cy = centro
    a, b = eixos[0] / 2, eixos[1] / 2  # semi-eixos
    
    # Transformar pontos para o sistema de coordenadas da elipse
    cos_a = np.cos(angulo_rad)
    sin_a = np.sin(angulo_rad)
    
    max_ratio = 0
    for ponto in pontos:
        px, py = ponto[0], ponto[1]
        
        # Transladar para origem
        dx = px - cx
        dy = py - cy
        
        # Rotacionar para alinhar com os eixos da elipse
        x_rot = dx * cos_a + dy * sin_a
        y_rot = -dx * sin_a + dy * cos_a
        
        # Calcular a razão em relação à elipse
        # Um ponto está dentro da elipse se (x/a)² + (y/b)² <= 1
        ratio = (x_rot / a) ** 2 + (y_rot / b) ** 2
        max_ratio = max(max_ratio, ratio)
    
    # Se max_ratio > 1, alguns pontos estão fora da elipse
    # Escalar os eixos pelo fator necessário
    if max_ratio > 1:
        fator_escala = np.sqrt(max_ratio) * 1.01  # 1% de margem adicional
        novos_eixos = (eixos[0] * fator_escala, eixos[1] * fator_escala)
    else:
        novos_eixos = eixos
    
    elipse_expandida = (centro, novos_eixos, angulo)
    return elipse_expandida


def desenhar_elipses_objetos(imagem_original, rotulos, num_objetos):
    """
    Desenha elipses de área mínima ao redor de cada objeto.
    
    Args:
        imagem_original: Imagem em escala de cinza original
        rotulos: Matriz com rótulos dos objetos
        num_objetos: Número total de objetos
    
    Returns:
        tuple: (imagem_com_elipses, lista_elipses)
    """
    # Converter para imagem colorida para desenhar elipses
    imagem_colorida = cv2.cvtColor(imagem_original, cv2.COLOR_GRAY2BGR)
    
    elipses_info = []
    
    for i in range(1, num_objetos + 1):
        # Encontrar coordenadas de todos os pixels do objeto
        coords = np.column_stack(np.where(rotulos == i))
        
        # OpenCV usa (x, y), então invertemos (linha, coluna) para (coluna, linha)
        pontos = np.array([[c[1], c[0]] for c in coords], dtype=np.float32)
        
        if len(pontos) >= 5:
            # Ajustar elipse de área mínima
            elipse = ajustar_elipse_minima(pontos)
            
            # Desenhar elipse
            cv2.ellipse(imagem_colorida, elipse, (0, 255, 0), 2)
            
            # Adicionar número do objeto
            centro = tuple(map(int, elipse[0]))
            cv2.putText(imagem_colorida, str(i), centro, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Armazenar informações da elipse
            area_elipse = np.pi * (elipse[1][0] / 2) * (elipse[1][1] / 2)
            elipses_info.append({
                'objeto': i,
                'centro': elipse[0],
                'eixo_maior': elipse[1][0],
                'eixo_menor': elipse[1][1],
                'angulo': elipse[2],
                'area_elipse': area_elipse
            })
        else:
            # Objeto muito pequeno para ajustar elipse
            elipses_info.append({
                'objeto': i,
                'centro': None,
                'eixo_maior': None,
                'eixo_menor': None,
                'angulo': None,
                'area_elipse': None,
                'nota': 'Objeto com menos de 5 pixels'
            })
    
    return imagem_colorida, elipses_info


def salvar_tabela_csv(pixels_por_objeto, elipses_info, nome_arquivo="analise_objetos.csv", densidade_px_por_mm=10.0):
    """
    Salva a tabela de análise em um arquivo CSV.
    
    Args:
        pixels_por_objeto: Lista com número de pixels por objeto
        elipses_info: Lista com informações das elipses
        nome_arquivo: Nome do arquivo CSV a ser criado
        densidade_px_por_mm: Densidade de pixels por milímetro (padrão: 10.0 px/mm)
    """
    # Calcular fator de conversão
    escala_px_para_mm = 1.0 / densidade_px_por_mm
    
    with open(nome_arquivo, 'w', newline='', encoding='utf-8') as arquivo_csv:
        writer = csv.writer(arquivo_csv)
        
        # Escrever cabeçalho com unidades em pixels e SI
        writer.writerow([
            'Objeto', 
            'Pixels (px)', 'Area_Objeto (mm²)',
            'Centro_X (px)', 'Centro_X (mm)',
            'Centro_Y (px)', 'Centro_Y (mm)',
            'Eixo_Maior (px)', 'Eixo_Maior (mm)',
            'Eixo_Menor (px)', 'Eixo_Menor (mm)',
            'Angulo (graus)', 
            'Area_Elipse (px²)', 'Area_Elipse (mm²)',
            'Razao (adimensional)', 
            'Observacao'
        ])
        
        # Escrever dados
        for i, (pixels, info) in enumerate(zip(pixels_por_objeto, elipses_info), 1):
            if info['centro'] is not None:
                # Conversões para SI (mm e mm²)
                area_objeto_mm2 = pixels * (escala_px_para_mm ** 2)
                centro_x_mm = info['centro'][0] * escala_px_para_mm
                centro_y_mm = info['centro'][1] * escala_px_para_mm
                eixo_maior_mm = info['eixo_maior'] * escala_px_para_mm
                eixo_menor_mm = info['eixo_menor'] * escala_px_para_mm
                area_elipse_mm2 = info['area_elipse'] * (escala_px_para_mm ** 2)
                
                writer.writerow([
                    i,
                    pixels,
                    f"{area_objeto_mm2:.4f}",
                    f"{info['centro'][0]:.2f}",
                    f"{centro_x_mm:.4f}",
                    f"{info['centro'][1]:.2f}",
                    f"{centro_y_mm:.4f}",
                    f"{info['eixo_maior']:.2f}",
                    f"{eixo_maior_mm:.4f}",
                    f"{info['eixo_menor']:.2f}",
                    f"{eixo_menor_mm:.4f}",
                    f"{info['angulo']:.2f}",
                    f"{info['area_elipse']:.2f}",
                    f"{area_elipse_mm2:.4f}",
                    f"{pixels / info['area_elipse']:.4f}",
                    ""
                ])
            else:
                nota = info.get('nota', 'Não foi possível ajustar elipse')
                writer.writerow([i, pixels, '', '', '', '', '', '', '', '', '', '', '', '', '', nota])


def imprimir_resultados(num_objetos, pixels_por_objeto, elipses_info):
    """
    Imprime os resultados da análise de forma formatada em tabela.
    """
    print("=" * 120)
    print("ANÁLISE DE OBJETOS PRETOS COM ELIPSES DE ÁREA MÍNIMA")
    print("=" * 120)
    print(f"\nQuantidade total de objetos identificados: {num_objetos}")
    print("\n" + "-" * 120)
    print("Informações detalhadas por objeto:")
    print("-" * 120)
    
    # Cabeçalho da tabela
    print(f"{'Objeto':>7} | {'Pixels':>7} | {'Centro (x, y)':>18} | {'Eixo Maior':>12} | {'Eixo Menor':>12} | {'Ângulo':>8} | {'Área Elipse':>13} | {'Razão':>7}")
    print("-" * 120)
    
    # Linhas da tabela
    for i, (pixels, info) in enumerate(zip(pixels_por_objeto, elipses_info), 1):
        if info['centro'] is not None:
            centro_str = f"({info['centro'][0]:.1f}, {info['centro'][1]:.1f})"
            eixo_maior = f"{info['eixo_maior']:.2f}"
            eixo_menor = f"{info['eixo_menor']:.2f}"
            angulo = f"{info['angulo']:.2f}°"
            area = f"{info['area_elipse']:.2f}"
            razao = f"{pixels / info['area_elipse']:.4f}"
            
            print(f"{i:>7} | {pixels:>7} | {centro_str:>18} | {eixo_maior:>12} | {eixo_menor:>12} | {angulo:>8} | {area:>13} | {razao:>7}")
        else:
            nota = info.get('nota', 'N/A')
            print(f"{i:>7} | {pixels:>7} | {'N/A':>18} | {'N/A':>12} | {'N/A':>12} | {'N/A':>8} | {'N/A':>13} | {'N/A':>7}  ({nota})")
    
    print("-" * 120)
    
    # Resumo estatístico
    print("\n" + "=" * 120)
    print("RESUMO ESTATÍSTICO")
    print("=" * 120)
    if pixels_por_objeto:
        print(f"Menor objeto: {min(pixels_por_objeto)} pixels")
        print(f"Maior objeto: {max(pixels_por_objeto)} pixels")
        print(f"Média de pixels: {np.mean(pixels_por_objeto):.2f} pixels")
        print(f"Total de pixels pretos: {sum(pixels_por_objeto)} pixels")
        
        areas_validas = [e['area_elipse'] for e in elipses_info if e['area_elipse'] is not None]
        if areas_validas:
            print(f"\nMenor área de elipse: {min(areas_validas):.2f} pixels²")
            print(f"Maior área de elipse: {max(areas_validas):.2f} pixels²")
            print(f"Média de área de elipse: {np.mean(areas_validas):.2f} pixels²")
    print("=" * 120)


# Exemplo de uso
if __name__ == "__main__":
    # Substitua pelo caminho da sua imagem
    caminho = "/home/tfk/Desktop/Teste_ImageJ/Imagens/vug_config_R1.91mm_N225_D3.82mm_AFA48F.tif"
    
    try:
        # Analisar objetos
        num_objetos, pixels_por_objeto, rotulos, imagem = analisar_objetos_pretos(caminho)
        
        # Desenhar elipses de área mínima
        imagem_com_elipses, elipses_info = desenhar_elipses_objetos(imagem, rotulos, num_objetos)
        
        # Imprimir resultados
        imprimir_resultados(num_objetos, pixels_por_objeto, elipses_info)
        
        # Salvar tabela em CSV
        # IMPORTANTE: Ajuste o valor de densidade_px_por_mm conforme sua calibração
        # Exemplo: se você tem 20 pixels por milímetro, use densidade_px_por_mm=20
        densidade_px_por_mm = 10.0  # Valor padrão: 10 pixels/mm
        salvar_tabela_csv(pixels_por_objeto, elipses_info, densidade_px_por_mm=densidade_px_por_mm)
        print(f"\nTabela salva como 'analise_objetos.csv' (densidade: {densidade_px_por_mm} px/mm)")
        
        # Salvar imagens
        cv2.imwrite("objetos_com_elipses.png", imagem_com_elipses)
        print("Imagem com elipses salva como 'objetos_com_elipses.png'")
        
        # Opcional: Criar visualização com objetos coloridos também
        imagem_colorida = np.zeros((*rotulos.shape, 3), dtype=np.uint8)
        for i in range(1, num_objetos + 1):
            cor = np.random.randint(0, 255, 3).tolist()
            imagem_colorida[rotulos == i] = cor
        cv2.imwrite("objetos_coloridos.png", imagem_colorida)
        print("Imagem com objetos coloridos salva como 'objetos_coloridos.png'")
        
    except Exception as e:
        print(f"Erro ao processar imagem: {e}")