import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Carregar resultados
results_dir = Path("/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/Arapua24/results/data")
data = np.loadtxt(results_dir / "results_summary.txt", delimiter=',', skiprows=1)

# Extrair colunas
time = data[:, 0]
dt = data[:, 1]
Qo = data[:, 2]
Qw = data[:, 3]
pin = data[:, 4]
pout = data[:, 5]
Vinj = data[:, 6]
Sin = data[:, 7]
Sout = data[:, 8]
Sdx = data[:, 9]

# Criar gráficos
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Saturação média vs tempo
axes[0, 0].plot(time, Sdx, 'b-', linewidth=2)
axes[0, 0].set_xlabel('Tempo [s]')
axes[0, 0].set_ylabel('Saturação média [-]')
axes[0, 0].set_title('Evolução da Saturação Média')
axes[0, 0].grid(True, alpha=0.3)

# Saturação entrada/saída
axes[0, 1].plot(time, Sin, 'r-', label='Entrada', linewidth=2)
axes[0, 1].plot(time, Sout, 'b-', label='Saída', linewidth=2)
axes[0, 1].set_xlabel('Tempo [s]')
axes[0, 1].set_ylabel('Saturação [-]')
axes[0, 1].set_title('Saturação Entrada/Saída')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Vazões
axes[0, 2].plot(time, Qo, 'r-', label='Óleo', linewidth=2)
axes[0, 2].plot(time, Qw, 'b-', label='Água', linewidth=2)
axes[0, 2].set_xlabel('Tempo [s]')
axes[0, 2].set_ylabel('Vazão [m³/s]')
axes[0, 2].set_title('Vazões de Óleo e Água')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].set_yscale('log')

# Pressão de entrada
axes[1, 0].plot(time, pin/1e5, 'g-', linewidth=2)
axes[1, 0].set_xlabel('Tempo [s]')
axes[1, 0].set_ylabel('Pressão [bar]')
axes[1, 0].set_title('Pressão de Entrada')
axes[1, 0].grid(True, alpha=0.3)

# Volume injetado acumulado
axes[1, 1].plot(time, Vinj, 'purple', linewidth=2)
axes[1, 1].set_xlabel('Tempo [s]')
axes[1, 1].set_ylabel('Volume injetado [m³]')
axes[1, 1].set_title('Volume Acumulado Injetado')
axes[1, 1].grid(True, alpha=0.3)

# Fração de água (water cut)
water_cut = Qw / (Qo + Qw + 1e-15)
axes[1, 2].plot(time, water_cut, 'cyan', linewidth=2)
axes[1, 2].set_xlabel('Tempo [s]')
axes[1, 2].set_ylabel('Water Cut [-]')
axes[1, 2].set_title('Fração de Água na Produção')
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].set_ylim([0, 1])

plt.tight_layout()
plt.savefig(results_dir.parent / 'analise_resultados.png', dpi=300)
plt.show()

# Estatísticas finais
print("\n" + "="*60)
print("ESTATÍSTICAS DA SIMULAÇÃO")
print("="*60)
print(f"Tempo total de simulação: {time[-1]:.2f} s")
print(f"Número de passos: {len(time)}")
print(f"Saturação média final: {Sdx[-1]:.4f}")
print(f"Saturação de saída final: {Sout[-1]:.4f}")
print(f"Volume total injetado: {Vinj[-1]:.6e} m³")
print(f"Pressão de entrada final: {pin[-1]/1e5:.2f} bar")
print(f"Water cut final: {water_cut[-1]:.4f}")
print(f"Recuperação de óleo: {(Vinj[-1] * Sdx[-1]):.6e} m³")
print("="*60)