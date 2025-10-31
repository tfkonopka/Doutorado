import cv2
import numpy as np
from scipy import ndimage

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


def imprimir_resultados(num_objetos, pixels_por_objeto, elipses_info):
    """
    Imprime os resultados da análise de forma formatada.
    """
    print("=" * 70)
    print("ANÁLISE DE OBJETOS PRETOS COM ELIPSES DE ÁREA MÍNIMA")
    print("=" * 70)
    print(f"\nQuantidade total de objetos identificados: {num_objetos}")
    print("\n" + "-" * 70)
    print("Informações detalhadas por objeto:")
    print("-" * 70)
    
    for i, (pixels, info) in enumerate(zip(pixels_por_objeto, elipses_info), 1):
        print(f"\nObjeto {i}:")
        print(f"  Pixels: {pixels}")
        
        if info['centro'] is not None:
            print(f"  Centro da elipse: ({info['centro'][0]:.2f}, {info['centro'][1]:.2f})")
            print(f"  Eixo maior: {info['eixo_maior']:.2f} pixels")
            print(f"  Eixo menor: {info['eixo_menor']:.2f} pixels")
            print(f"  Ângulo: {info['angulo']:.2f}°")
            print(f"  Área da elipse: {info['area_elipse']:.2f} pixels²")
            print(f"  Razão (pixels objeto / área elipse): {pixels / info['area_elipse']:.4f}")
        else:
            print(f"  {info.get('nota', 'Não foi possível ajustar elipse')}")
    
    print("\n" + "=" * 70)
    print("RESUMO ESTATÍSTICO")
    print("=" * 70)
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
    print("=" * 70)


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
        
        # Salvar imagens
        cv2.imwrite("objetos_com_elipses.png", imagem_com_elipses)
        print("\nImagem com elipses salva como 'objetos_com_elipses.png'")
        
        # Opcional: Criar visualização com objetos coloridos também
        imagem_colorida = np.zeros((*rotulos.shape, 3), dtype=np.uint8)
        for i in range(1, num_objetos + 1):
            cor = np.random.randint(0, 255, 3).tolist()
            imagem_colorida[rotulos == i] = cor
        cv2.imwrite("objetos_coloridos.png", imagem_colorida)
        print("Imagem com objetos coloridos salva como 'objetos_coloridos.png'")
        
    except Exception as e:
        print(f"Erro ao processar imagem: {e}")