// Script ImageJ para identificação de clusters de círculos
// Exclui círculos com raio específico
// Autor: Script gerado para análise de clusters

// ===== CAMINHO DA IMAGEM =====
// Coloque aqui o caminho completo da sua imagem
// Exemplos:
//   Windows: "C:\\Users\\SeuNome\\Desktop\\imagem.jpg"
//   Mac/Linux: "/Users/SeuNome/Desktop/imagem.jpg"
caminhoImagem = "/home/tfk/Desktop/Teste_ImageJ/Imagens/vug_config_R1.91mm_N225_D3.82mm_AFA48F.tif";


// ===== CALIBRAÇÃO DA IMAGEM =====
// Defina a calibração espacial (quantos pixels = quantos mm)
pixelsPorMM = 10; // Exemplo: 10 pixels = 1 mm (ajuste conforme sua imagem)

// ===== PARÂMETROS CONFIGURÁVEIS =====
raioExcluir_mm = 5.0; // Raio em mm dos círculos a serem excluídos
toleranciaRaio_mm = 0.5; // Tolerância para exclusão (±)
distanciaCluster_mm = 2.0; // Distância máxima entre círculos para formar cluster
minCirculosPorCluster = 2; // Mínimo de círculos para considerar um cluster

// Parâmetros de detecção
minArea_mm2 = 0.5; // Área mínima dos círculos em mm²
maxArea_mm2 = 700; // Área máxima dos círculos em mm²
circularidadeMin = 0.5; // 0-1 (1 = círculo perfeito)

// ===== INÍCIO DO SCRIPT =====
print("\\Clear");
print("=== Análise de Clusters de Círculos ===\n");

// Abrir imagem
print("Abrindo imagem: " + caminhoImagem);
open(caminhoImagem);
img = getImageID();
titulo = getTitle();
print("Imagem carregada: " + titulo);

// Aplicar calibração
pixelWidth_mm = 1.0 / pixelsPorMM;
run("Set Scale...", "distance=" + pixelsPorMM + " known=1 unit=mm");
print("Calibração aplicada: " + pixelsPorMM + " pixels = 1 mm\n");

// Converter áreas de mm² para pixels²
minArea_px = minArea_mm2 * pixelsPorMM * pixelsPorMM;
maxArea_px = maxArea_mm2 * pixelsPorMM * pixelsPorMM;
distanciaCluster_px = distanciaCluster_mm * pixelsPorMM;

// Duplicar imagem para processamento
selectImage(img);
run("Duplicate...", "title=processamento");
imgProc = getImageID();

// Pré-processamento
print("Pré-processamento da imagem...");
run("8-bit");

// Verificar se fundo é branco ou preto
getRawStatistics(nPixels, mean);
if (mean > 127) {
    // Fundo branco, círculos pretos - inverter
    print("Detectado: círculos escuros em fundo claro");
    run("Invert");
} else {
    print("Detectado: círculos claros em fundo escuro");
}

run("Gaussian Blur...", "sigma=2");
setAutoThreshold("Default dark");
run("Convert to Mask");

// Limpar ruídos
run("Fill Holes");
run("Watershed");

// Detectar partículas (círculos)
print("\nDetectando círculos...");
run("Set Measurements...", "area mean centroid perimeter fit redirect=None decimal=3");
run("Analyze Particles...", "size=" + minArea_px + "-" + maxArea_px + 
    " circularity=" + circularidadeMin + "-1.00 show=Nothing display clear");

n = nResults;
print("Objetos detectados: " + n);

if (n == 0) {
    print("\nERRO: Nenhum círculo detectado!");
    print("Dicas:");
    print("- Verifique a calibração (pixelsPorMM)");
    print("- Ajuste minArea_mm2 e maxArea_mm2");
    print("- Reduza circularidadeMin se os objetos não são círculos perfeitos");
    selectImage(imgProc);
    close();
    exit();
}

// Armazenar dados dos círculos
xCentros = newArray(n);
yCentros = newArray(n);
raios = newArray(n);
areas = newArray(n);
valido = newArray(n);

for (i = 0; i < n; i++) {
    xCentros[i] = getResult("X", i);
    yCentros[i] = getResult("Y", i);
    areas[i] = getResult("Area", i);
    
    // Calcular raio a partir da área (assumindo círculo)
    raios[i] = sqrt(areas[i] / PI);
    raio_mm = raios[i] / pixelsPorMM;
    
    // Verificar se deve excluir por raio
    if (abs(raio_mm - raioExcluir_mm) <= toleranciaRaio_mm) {
        valido[i] = 0;
    } else {
        valido[i] = 1;
    }
}

// Contar círculos válidos
nValidos = 0;
for (i = 0; i < n; i++) {
    if (valido[i] == 1) nValidos++;
}

print("\nCírculos válidos (após filtro de raio): " + nValidos);
print("Círculos excluídos (raio " + raioExcluir_mm + "±" + toleranciaRaio_mm + " mm): " + (n - nValidos));

// ===== IDENTIFICAÇÃO DE CLUSTERS =====
print("\n=== Análise de Clusters ===");
print("Distância máxima para cluster: " + distanciaCluster_mm + " mm");

// Criar array de clusters (-1 = não atribuído)
cluster = newArray(n);
for (i = 0; i < n; i++) cluster[i] = -1;

numCluster = 0;

// Algoritmo de clustering por proximidade (DBSCAN simplificado)
for (i = 0; i < n; i++) {
    if (valido[i] == 0) continue;
    if (cluster[i] != -1) continue;
    
    // Iniciar novo cluster
    cluster[i] = numCluster;
    pilha = newArray(1);
    pilha[0] = i;
    topo = 0;
    
    // Expandir cluster
    while (topo >= 0) {
        atual = pilha[topo];
        topo--;
        
        for (j = 0; j < n; j++) {
            if (valido[j] == 0) continue;
            if (cluster[j] != -1) continue;
            
            // Calcular distância entre centros
            dx = xCentros[atual] - xCentros[j];
            dy = yCentros[atual] - yCentros[j];
            dist = sqrt(dx*dx + dy*dy);
            
            if (dist <= distanciaCluster_px) {
                cluster[j] = numCluster;
                topo++;
                // Expandir array se necessário
                if (topo >= pilha.length) {
                    pilha = Array.concat(pilha, j);
                } else {
                    pilha[topo] = j;
                }
            }
        }
    }
    
    numCluster++;
}

// Contar círculos por cluster
contagemCluster = newArray(numCluster);
Array.fill(contagemCluster, 0);
for (i = 0; i < n; i++) {
    if (cluster[i] >= 0) {
        contagemCluster[cluster[i]]++;
    }
}

// Filtrar clusters válidos
numClustersValidos = 0;
for (i = 0; i < numCluster; i++) {
    if (contagemCluster[i] >= minCirculosPorCluster) {
        numClustersValidos++;
    }
}

print("Total de agrupamentos encontrados: " + numCluster);
print("Clusters válidos (≥" + minCirculosPorCluster + " círculos): " + numClustersValidos);

// Estatísticas detalhadas dos clusters
print("\n=== Detalhes dos Clusters ===");
for (c = 0; c < numCluster; c++) {
    if (contagemCluster[c] >= minCirculosPorCluster) {
        print("Cluster " + (c+1) + ": " + contagemCluster[c] + " círculos");
    }
}

// ===== VISUALIZAÇÃO DOS RESULTADOS =====
print("\nGerando visualização...");
selectImage(img);
run("Duplicate...", "title=Resultado_Clusters");
run("RGB Color");

// Cores para clusters
cores = newArray("red", "green", "blue", "yellow", "magenta", "cyan", "orange", "pink", "white", "lime");
setLineWidth(2);

// Desenhar círculos
for (i = 0; i < n; i++) {
    x = xCentros[i];
    y = yCentros[i];
    r = raios[i];
    
    if (valido[i] == 0) {
        // Círculos excluídos em cinza tracejado
        setColor("gray");
        drawCircle(x, y, r);
        setFont("SansSerif", 10, "bold");
        drawString("X", x-4, y+4);
    } else if (cluster[i] >= 0 && contagemCluster[cluster[i]] >= minCirculosPorCluster) {
        // Círculos em clusters válidos - coloridos
        corIdx = cluster[i] % cores.length;
        setColor(cores[corIdx]);
        drawCircle(x, y, r);
        
        // Marcar centro
        fillOval(x-3, y-3, 6, 6);
        
        // Numerar cluster
        setFont("SansSerif", 12, "bold");
        drawString("" + (cluster[i]+1), x+r+5, y);
    } else {
        // Círculos isolados em amarelo
        setColor("yellow");
        drawCircle(x, y, r);
        fillOval(x-2, y-2, 4, 4);
    }
}

// Criar tabela de resultados dos clusters
run("Clear Results");
for (c = 0; c < numCluster; c++) {
    if (contagemCluster[c] < minCirculosPorCluster) continue;
    
    // Calcular centroide e propriedades do cluster
    xSum = 0; ySum = 0; count = 0;
    raioMedio = 0;
    raioMin = 999999;
    raioMax = 0;
    
    for (i = 0; i < n; i++) {
        if (cluster[i] == c) {
            xSum += xCentros[i];
            ySum += yCentros[i];
            raioMedio += raios[i];
            if (raios[i] < raioMin) raioMin = raios[i];
            if (raios[i] > raioMax) raioMax = raios[i];
            count++;
        }
    }
    
    xCentroide = xSum / count;
    yCentroide = ySum / count;
    raioMedio = raioMedio / count;
    
    setResult("Cluster", nResults, c + 1);
    setResult("N_Circulos", nResults-1, count);
    setResult("Centro_X_mm", nResults-1, xCentroide / pixelsPorMM);
    setResult("Centro_Y_mm", nResults-1, yCentroide / pixelsPorMM);
    setResult("Raio_Medio_mm", nResults-1, raioMedio / pixelsPorMM);
    setResult("Raio_Min_mm", nResults-1, raioMin / pixelsPorMM);
    setResult("Raio_Max_mm", nResults-1, raioMax / pixelsPorMM);
}

updateResults();

print("\n=== Análise Concluída ===");
print("Resultados exibidos na tabela Results");
print("Imagem com clusters coloridos: Resultado_Clusters");
print("\nLegenda:");
print("- Círculos coloridos numerados = Clusters válidos");
print("- Círculos amarelos = Isolados (não formam cluster)");
print("- Círculos cinza com X = Excluídos por raio");

// Limpar imagem de processamento
selectImage(imgProc);
close();

// Funções auxiliares
function drawCircle(x, y, r) {
    drawOval(x-r, y-r, 2*r, 2*r);
}

function abs(x) {
    if (x < 0) return -x;
    return x;
}