// ImageJ Macro - Gerador de Geometrias Vugulares
// Baseado nas especificações do paper sobre domínios vugulares sintéticos

// Parâmetros padrão do domínio
L = 200; // Comprimento em mm
W = 100; // Largura em mm
pixels_per_mm = 10; // Resolução (pixels por mm) - ajustável

// Conversão para pixels
L_pixels = L * pixels_per_mm;
W_pixels = W * pixels_per_mm;

// Interface para entrada de dados
Dialog.create("Gerador de Geometrias Vugulares");
Dialog.addMessage("=== PARÂMETROS DO DOMÍNIO ===");
Dialog.addNumber("Comprimento (mm):", L);
Dialog.addNumber("Largura (mm):", W);
Dialog.addNumber("Resolução (pixels/mm):", pixels_per_mm);

Dialog.addMessage("=== PARÂMETROS DOS VUGS ===");
Dialog.addChoice("Raio dos vugs (mm):", newArray("1.91", "2.91", "Custom"), "1.91");
Dialog.addNumber("Raio customizado (mm):", 2.0);
Dialog.addChoice("Número de inclusões:", newArray("225", "525", "Custom"), "225");
Dialog.addNumber("Número customizado:", 300);

Dialog.addMessage("=== CONFIGURAÇÃO DA GRADE ===");
Dialog.addChoice("Distância entre nós:", newArray("D = R", "D = 1.5R", "D = 2R", "D = 2R + δ", "D = 3R", "Custom"), "D = 2R");
Dialog.addNumber("Delta (δ) para D = 2R + δ:", 0.5);
Dialog.addNumber("Distância customizada (em função de R):", 2.0);

Dialog.addMessage("=== OPÇÕES DE SAÍDA ===");
Dialog.addNumber("Número de imagens a gerar:", 1);
Dialog.addCheckbox("Salvar imagem", true);
Dialog.addCheckbox("Calcular macroporosidade", true);
Dialog.addCheckbox("Mostrar estatísticas", true);
Dialog.addCheckbox("Salvar estatísticas em arquivo", true);
Dialog.addString("Prefixo do arquivo:", "vug_config");
Dialog.addString("Pasta de destino:", "/home/tfk/Desktop/Teste_ImageJ/Imagens");

Dialog.show();

// Leitura dos parâmetros
L = Dialog.getNumber();
W = Dialog.getNumber();
pixels_per_mm = Dialog.getNumber();
radius_choice = Dialog.getChoice();
custom_radius = Dialog.getNumber();
n_inclusions_choice = Dialog.getChoice();
custom_n_inclusions = Dialog.getNumber();
distance_choice = Dialog.getChoice();
delta = Dialog.getNumber();
custom_distance_factor = Dialog.getNumber();
num_images = Dialog.getNumber();
save_image = Dialog.getCheckbox();
calc_porosity = Dialog.getCheckbox();
show_stats = Dialog.getCheckbox();
save_stats_file = Dialog.getCheckbox();
file_prefix = Dialog.getString();
output_dir_input = Dialog.getString();

// Verificar e definir diretório de saída
output_dir = "";
if (save_image || save_stats_file) {
    if (output_dir_input != "") {
        output_dir = output_dir_input;
        if (!endsWith(output_dir, File.separator)) {
            output_dir = output_dir + File.separator;
        }
        print("Usando pasta informada: " + output_dir);
    } else {
        output_dir = getDirectory("Escolha a pasta para salvar os arquivos");
        print("Pasta selecionada via diálogo: " + output_dir);
    }

    if (output_dir == "") {
        print("AVISO: Nenhuma pasta selecionada - arquivos não serão salvos");
        save_image = false;
        save_stats_file = false;
    }
}

// Validar número de imagens
if (num_images < 1) {
    num_images = 1;
    print("Número de imagens ajustado para: " + num_images);
}

print("=== INICIANDO GERAÇÃO DE " + num_images + " IMAGEM(NS) ===");

// Loop principal para gerar múltiplas imagens
for (img_count = 1; img_count <= num_images; img_count++) {

    print("\n=== GERANDO IMAGEM " + img_count + " DE " + num_images + " ===");

    // Conversão para pixels
    L_pixels = L * pixels_per_mm;
    W_pixels = W * pixels_per_mm;

    // Definir raio
    if (radius_choice == "Custom") {
        R = custom_radius;
    } else {
        R = parseFloat(radius_choice);
    }
    R_pixels = R * pixels_per_mm;

    // Definir número de inclusões
    if (n_inclusions_choice == "Custom") {
        n_inclusions = custom_n_inclusions;
    } else {
        n_inclusions = parseInt(n_inclusions_choice);
    }

    // Definir distância entre nós da grade
    if (distance_choice == "D = R") {
        D = R;
    } else if (distance_choice == "D = 1.5R") {
        D = 1.5 * R;
    } else if (distance_choice == "D = 2R") {
        D = 2 * R;
    } else if (distance_choice == "D = 2R + δ") {
        D = 2 * R + delta;
    } else if (distance_choice == "D = 3R") {
        D = 3 * R;
    } else {
        D = custom_distance_factor * R;
    }
    D_pixels = D * pixels_per_mm;

    // Criar imagem base
    newImage("Vugular_Geometry_" + img_count, "8-bit white", L_pixels, W_pixels, 1);
    image_id = getImageID();

    // Desenhar borda preta no perímetro
    setForegroundColor(0, 0, 0); // Preto
    setLineWidth(1);

    // Borda superior
    //drawLine(0, 0, L_pixels-1, 0);
    // Borda inferior  
    //drawLine(0, W_pixels-1, L_pixels-1, W_pixels-1);
    // Borda esquerda
    //drawLine(0, 0, 0, W_pixels-1);
    // Borda direita
    //drawLine(L_pixels-1, 0, L_pixels-1, W_pixels-1);

    // Criar grade de pontos candidatos
    grid_points_x = newArray();
    grid_points_y = newArray();
    point_count = 0;

    // Margem mínima de 2mm das bordas
    border_margin = 2; // mm
    border_margin_pixels = border_margin * pixels_per_mm;

    // Definir limites para todas as 4 bordas
    // Distância mínima da borda esquerda e superior
    min_x = border_margin_pixels + R_pixels;
    min_y = border_margin_pixels + R_pixels;

    // Distância máxima das bordas direita e inferior
    max_x = L_pixels - border_margin_pixels - R_pixels;
    max_y = W_pixels - border_margin_pixels - R_pixels;

    // Verificar se os limites são válidos
    if (max_x <= min_x || max_y <= min_y) {
        showMessage("Erro", "Domínio muito pequeno para os parâmetros escolhidos!\n" +
                   "Reduza o raio dos vugs ou aumente o tamanho do domínio.");
        exit();
    }

    for (x = min_x; x <= max_x; x += D_pixels) {
        for (y = min_y; y <= max_y; y += D_pixels) {
            grid_points_x[point_count] = x;
            grid_points_y[point_count] = y;
            point_count++;
        }
    }

    print("Total de pontos na grade: " + point_count);

    // Seleção aleatória de pontos para colocar as inclusões
    if (n_inclusions > point_count) {
        n_inclusions = point_count;
        print("Número de inclusões ajustado para: " + n_inclusions);
    }

    selected_indices = newArray(n_inclusions);
    used_indices = newArray(point_count);

    // Inicializar array de índices usados
    for (i = 0; i < point_count; i++) {
        used_indices[i] = false;
    }

    // Seleção aleatória sem repetição
    for (i = 0; i < n_inclusions; i++) {
        do {
            random_index = floor(random() * point_count);
        } while (used_indices[random_index]);

        selected_indices[i] = random_index;
        used_indices[random_index] = true;
    }

    // Desenhar as inclusões circulares
    setForegroundColor(0, 0, 0); // preto para os vugs
    for (i = 0; i < n_inclusions; i++) {
        index = selected_indices[i];
        x_center = grid_points_x[index];
        y_center = grid_points_y[index];

        // Criar seleção circular
        makeOval(x_center - R_pixels, y_center - R_pixels, 2 * R_pixels, 2 * R_pixels);
        fill();
    }

    // Remover seleção
    run("Select None");

    // Preparar string com todas as estatísticas para arquivo
    stats_content = "";
    stats_content += "=== CONFIGURAÇÃO GERADA ===\n";
    stats_content += "Imagem: " + img_count + " de " + num_images + "\n";
    stats_content += "Data/Hora: " + getTime() + "\n";
    stats_content += "Arquivo: " + file_prefix + "\n\n";

    stats_content += "=== PARÂMETROS DO DOMÍNIO ===\n";
    stats_content += "Comprimento: " + L + " mm (" + L_pixels + " pixels)\n";
    stats_content += "Largura: " + W + " mm (" + W_pixels + " pixels)\n";
    stats_content += "Resolução: " + pixels_per_mm + " pixels/mm\n\n";

    stats_content += "=== PARÂMETROS DOS VUGS ===\n";
    stats_content += "Raio: " + R + " mm (" + R_pixels + " pixels)\n";
    stats_content += "Número de inclusões: " + n_inclusions + "\n";
    stats_content += "Distância entre nós: " + D + " mm (" + D_pixels + " pixels)\n";
    stats_content += "Total de pontos na grade: " + point_count + "\n\n";

    // ===== SEÇÃO CORRIGIDA - CÁLCULO DA MACROPOROSIDADE =====
    if (calc_porosity) {
        selectImage(image_id);
        
        // Contar apenas pixels pretos dos vugs, excluindo as bordas
        vug_pixels = 0;
        border_pixels = 0;
        
        // Função para verificar se um pixel está na borda
        function isOnBorder(x, y, width, height) {
            return (x == 0 || x == width-1 || y == 0 || y == height-1);
        }
        
        // Percorrer todos os pixels
        for (x = 0; x < L_pixels; x++) {
            for (y = 0; y < W_pixels; y++) {
                pixel_value = getPixel(x, y);
                if (pixel_value == 0) { // Pixel preto
                    if (isOnBorder(x, y, L_pixels, W_pixels)) {
                        border_pixels++; // Pixel da borda
                    } else {
                        vug_pixels++; // Pixel de vug
                    }
                }
            }
        }
        
        // Área total incluindo bordas
        total_pixels = L_pixels * W_pixels;
        
        // Macroporosidade = pixels dos vugs / área total
        macroporosity = vug_pixels / total_pixels;
        
        stats_content += "=== ESTATÍSTICAS REAIS ===\n";
        stats_content += "Pixels pretos (vugs): " + vug_pixels + "\n";
        stats_content += "Pixels pretos (bordas): " + border_pixels + "\n";
        stats_content += "Total de pixels pretos: " + (vug_pixels + border_pixels) + "\n";
        stats_content += "Área total: " + total_pixels + " pixels\n";
        stats_content += "Macroporosidade: " + d2s(macroporosity, 4) + "\n";
        stats_content += "Macroporosidade (%): " + d2s(macroporosity * 100, 2) + "%\n\n";

        print("Pixels de vugs: " + vug_pixels);
        print("Pixels de bordas: " + border_pixels);
        print("Macroporosidade: " + d2s(macroporosity * 100, 2) + "%");
    }

    // Mostrar estatísticas adicionais se solicitado
    if (show_stats) {
        theoretical_vug_area = PI * R * R * n_inclusions;
        domain_area = L * W;
        theoretical_porosity = theoretical_vug_area / domain_area;

        stats_content += "=== PARÂMETROS TEÓRICOS ===\n";
        stats_content += "Área teórica dos vugs: " + d2s(theoretical_vug_area, 2) + " mm²\n";
        stats_content += "Área do domínio: " + domain_area + " mm²\n";
        stats_content += "Porosidade teórica: " + d2s(theoretical_porosity * 100, 2) + "%\n\n";

        if (calc_porosity) {
            overlap_factor = theoretical_porosity - macroporosity;
            stats_content += "=== ANÁLISE DE SOBREPOSIÇÃO ===\n";
            stats_content += "Fator de sobreposição: " + d2s(overlap_factor * 100, 2) + "%\n";
            stats_content += "Eficiência de preenchimento: " + d2s((macroporosity/theoretical_porosity) * 100, 2) + "%\n\n";
        }
    }

    // Gerar número hexadecimal aleatório de 6 dígitos
    hex_id = "";
    for (i = 0; i < 6; i++) {
        random_digit = floor(random() * 16);
        if (random_digit < 10) {
            hex_id += toString(random_digit);
        } else if (random_digit == 10) {
            hex_id += "A";
        } else if (random_digit == 11) {
            hex_id += "B";
        } else if (random_digit == 12) {
            hex_id += "C";
        } else if (random_digit == 13) {
            hex_id += "D";
        } else if (random_digit == 14) {
            hex_id += "E";
        } else {
            hex_id += "F";
        }
    }

    stats_content += "=== INFORMAÇÕES DO ARQUIVO ===\n";
    stats_content += "ID: " + hex_id + "\n";
    stats_content += "Configuração: R=" + R + "mm, N=" + n_inclusions + ", D=" + D + "mm\n";

    // Salvar arquivos se solicitado
    base_filename = file_prefix + "_R" + d2s(R, 2) + "mm_N" + n_inclusions + "_D" + d2s(D, 2) + "mm_" + hex_id;

    if (save_image && output_dir != "") {
        // Garantir que a imagem dos vugs está selecionada
        selectImage(image_id);

        image_filename = base_filename + ".tif";
        full_path = output_dir + image_filename;

        // Tentar salvar
        saveAs("Tiff", full_path);

        // Verificar se foi salvo
        if (File.exists(full_path)) {
            print("✓ Imagem " + img_count + " salva: " + image_filename);
        } else {
            print("✗ Erro ao salvar imagem " + img_count);
        }
    }

    if (save_stats_file && output_dir != "") {
        stats_filename = base_filename + "_stats.txt";
        full_path_stats = output_dir + stats_filename;

        // Tentar salvar
        File.saveString(stats_content, full_path_stats);

        // Verificar se foi salvo
        if (File.exists(full_path_stats)) {
            print("✓ Estatísticas " + img_count + " salvas: " + stats_filename);
        } else {
            print("✗ Erro ao salvar estatísticas " + img_count);
        }
    }

} // Fim do loop principal


print("\n=== GERAÇÃO CONCLUÍDA ===");
print("Total de imagens geradas: " + num_images);
print("Configuração base: R=" + R + "mm, N=" + n_inclusions + ", D=" + D + "mm");

// Função auxiliar para gerar múltiplas configurações
function generateMultipleConfigurations() {
    radii = newArray(1.91, 2.91);
    n_vugs = newArray(225, 525);
    distances = newArray("R", "1.5R", "2R", "2R+0.5", "3R");

    config_count = 0;
    for (r = 0; r < radii.length; r++) {
        for (n = 0; n < n_vugs.length; n++) {
            for (d = 0; d < distances.length; d++) {
                config_count++;
                print("Configuração " + config_count + ": R=" + radii[r] + ", N=" + n_vugs[n] + ", D=" + distances[d]);
                // Aqui você pode chamar a função principal com diferentes parâmetros
            }
        }
    }
}
