"""
Biblioteca para simulação de fluxo bifásico usando método Brinkman-IMPES
Versão Melhorada - Com métodos robustos de passo de tempo adaptativo para DG
"""

from fenics import *
import time
import ufl_legacy as ufl
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

# ==================== CONSTANTES ====================

class PhysicalConstants:
    """Constantes físicas utilizadas nas simulações"""
    MILLIDARCY_TO_M2 = 9.86923e-16
    KGF_CM2_TO_PA = 98066.5

# ==================== EXPRESSÕES CUSTOMIZADAS ====================

class PiecewiseConstant(UserExpression):
    """Expressão constante por partes baseada em marcadores de células"""

    def __init__(self, values: Dict[int, float], markers, **kwargs):
        self._values = values
        self._markers = markers
        super().__init__(**kwargs)

    def eval_cell(self, values, x, cell):
        values[0] = self._values[self._markers[cell.index]]

    def value_shape(self):
        return tuple()

# ==================== FUNÇÕES AUXILIARES ====================

class FlowEquations:
    """Equações constitutivas para fluxo bifásico"""

    @staticmethod
    def tensor_jump(v, n):
        """Salto do tensor através de interfaces"""
        return ufl.outer(v, n)("+") + ufl.outer(v, n)("-")

    @staticmethod
    def lambda_inv(s, mu_w: float, mu_o: float, no: float, nw: float):
        """Inverso da mobilidade total"""
        return 1.0 / ((s**nw) / mu_w + ((1.0 - s)**no) / mu_o)

    @staticmethod
    def fractional_flow(s, mu_rel: float, no, nw):
        """Função de fluxo fracionário (Buckley-Leverett)"""
        return s**nw / (s**nw + mu_rel * (1.0 - s)**no)

    @staticmethod
    def fractional_flow_vugg(s):
        """Fluxo fracionário em vuggy (cavidades)"""
        return s

    @staticmethod
    def mu_brinkman(s, mu_o: float, mu_w: float):
        """Viscosidade efetiva pelo modelo de Brinkman"""
        return s * mu_w + (1.0 - s) * mu_o

# ==================== I/O ====================

class DataWriter:
    """Gerenciamento de escrita de dados de simulação"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.output_dir / "results.txt"
        self.summary_file = self.output_dir / "results_summary.txt"

    def initialize_file(self):
        """Inicializa arquivo de resultados com cabeçalho"""
        headers = ["time", "dt", "Qo", "Qw", "pin", "pout", 
                   "Vinj", "Sin", "Sout", "Sdx"]

        with open(self.results_file, 'w') as f:
            f.write(','.join(headers) + '\n')

        print(f"Arquivo inicializado: {self.results_file}")

    def append_timestep(self, data: Dict[str, float]):
        """Adiciona dados de um passo de tempo"""
        keys = ["time", "dt", "Qo", "Qw", "pin", "pout", 
                "Vinj", "Sin", "Sout", "Sdx"]

        with open(self.results_file, 'a') as f:
            row = ','.join(str(data[key]) for key in keys)
            f.write(row + '\n')

    def write_summary(self, data_arrays: Dict[str, list]):
        """Escreve resumo completo da simulação"""
        if not data_arrays or len(data_arrays.get('time', [])) == 0:
            print("Aviso: Nenhum dado para gravar no summary")
            return

        keys = ["time", "dt", "Qo", "Qw", "pin", "pout", 
                "Vinj", "Sin", "Sout", "Sdx"]

        with open(self.summary_file, 'w') as f:
            f.write(','.join(keys) + '\n')

            n_steps = len(data_arrays['time'])
            for i in range(n_steps):
                row = ','.join(str(data_arrays[key][i]) for key in keys)
                f.write(row + '\n')

        print(f"Summary gravado em: {self.summary_file}")
        print(f"Total de {n_steps} passos de tempo salvos")

# ==================== SOLVER PRINCIPAL ====================

class BrinkmanIMPESSolver:
    """
    Solver para simulação de fluxo bifásico óleo-água usando
    método IMPES (Implicit Pressure Explicit Saturation) com
    equação de Brinkman para meios vuggy
    """

    def __init__(self, 
                 mesh_dir: str,
                 output_dir: str,
                 mu_w: float = 1e-3,
                 mu_o: float = 1e-3,
                 perm_darcy: float = 1.0,
                 dt: float = 0.1,
                 phi: float = 0.2,
                 inlet_velocity: float = 1.0e-6,
                 pin: float = None,
                 pout: float = None,
                 adaptive_dt: bool = False,
                 CFL_max: float = 0.5,
                 dt_min: float = 1e-6,
                 dt_max: float = 10.0,
                 checkpoint_interval: int = 0,
                 dt_method: str = "hybrid"):
        """
        Inicializa o solver

        Args:
            mesh_dir: Diretório contendo mesh/domains/boundaries
            output_dir: Diretório para saída de resultados
            mu_w: Viscosidade da água [Pa.s]
            mu_o: Viscosidade do óleo [Pa.s]
            perm_darcy: Permeabilidade em Darcy
            dt: Passo de tempo inicial [s]
            phi: Porosidade [-]
            inlet_velocity: Velocidade na entrada [m/s] (componente x)
            pin: Pressão de entrada [Pa] (se None, usa 2*KGF_CM2_TO_PA)
            pout: Pressão de saída [Pa] (se None, usa KGF_CM2_TO_PA)
            adaptive_dt: Se True, usa passo de tempo adaptativo
            CFL_max: Número de CFL máximo para dt adaptativo
            dt_min: Passo de tempo mínimo [s]
            dt_max: Passo de tempo máximo [s]
            checkpoint_interval: Salva checkpoint a cada N passos (0 = desativado)
            dt_method: Método de cálculo de dt ("classic", "local", "saturation", "hybrid")
        """
        self.mesh_dir = Path(mesh_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Propriedades físicas
        self.mu_w = mu_w
        self.mu_o = mu_o
        self.mu_rel = mu_w / mu_o
        self.phi = phi
        self.k_matrix = perm_darcy * PhysicalConstants.MILLIDARCY_TO_M2

        # Condições de contorno
        self.inlet_velocity = inlet_velocity
        self.pin_value = pin if pin is not None else 2 * PhysicalConstants.KGF_CM2_TO_PA
        self.pout_value = pout if pout is not None else PhysicalConstants.KGF_CM2_TO_PA

        # Parâmetros numéricos
        self.dt = dt
        self.dt_initial = dt
        self.adaptive_dt = adaptive_dt
        self.CFL_max = CFL_max
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_method = dt_method
        self.checkpoint_interval = checkpoint_interval
        self.t = 0.0
        self.step = 0

        # Histórico para suavização temporal
        self.dt_history = []

        # Componentes FEniCS (inicializados em setup)
        self.mesh = None
        self.W = None
        self.R = None
        self.U = None
        self.S = None

        # Escritor de dados
        self.writer = DataWriter(self.output_dir / "data")

        # Histórico de resultados
        self.results = {
            'time': [],
            'dt': [],
            'Qo': [],
            'Qw': [],
            'pin': [],
            'pout': [],
            'Vinj': [],
            'Sin': [],
            'Sout': [],
            'Sdx': []
        }

    def load_mesh(self):
        """Carrega mesh e marcadores"""
        print("Carregando mesh...")
        self.mesh = Mesh()
        with XDMFFile(str(self.mesh_dir / "mesh.xdmf")) as infile:
            infile.read(self.mesh)

        mvc = MeshValueCollection("size_t", self.mesh, 2)
        with XDMFFile(str(self.mesh_dir / "domains.xdmf")) as infile:
            infile.read(mvc)
        self.markers = cpp.mesh.MeshFunctionSizet(self.mesh, mvc)

        mvc = MeshValueCollection("size_t", self.mesh, 1)
        with XDMFFile(str(self.mesh_dir / "boundaries.xdmf")) as infile:
            infile.read(mvc)
        self.boundaries = cpp.mesh.MeshFunctionSizet(self.mesh, mvc)
        print("Mesh carregada com sucesso")

    def setup_function_spaces(self, order: int = 1):
        """Define espaços de funções"""
        print("Configurando espaços de funções...")
        V = FiniteElement("BDM", self.mesh.ufl_cell(), order)
        Q = FiniteElement("DG", self.mesh.ufl_cell(), order - 1)

        self.W = FunctionSpace(self.mesh, V * Q)
        self.R = FunctionSpace(self.mesh, "DG", order - 1)

        self.U = Function(self.W)
        self.S = Function(self.R)
        self.s0 = Function(self.R)
        self.s0.vector()[:] = 0.0
        print("Espaços de funções configurados")

    def setup_boundary_conditions(self):
        """Define condições de contorno"""
        print("Configurando condições de contorno...")
        print(f"  - Velocidade de entrada: {self.inlet_velocity} m/s")
        print(f"  - Pressão de entrada: {self.pin_value:.2f} Pa ({self.pin_value/PhysicalConstants.KGF_CM2_TO_PA:.2f} kgf/cm²)")
        print(f"  - Pressão de saída: {self.pout_value:.2f} Pa ({self.pout_value/PhysicalConstants.KGF_CM2_TO_PA:.2f} kgf/cm²)")

        self.pin_bc = self.pin_value
        self.pout_bc = self.pout_value

        # BC1: Velocidade prescrita na entrada (boundary 1)
        bc1 = DirichletBC(self.W.sub(0), 
                         Constant((self.inlet_velocity, 0.0)), 
                         self.boundaries, 1)

        # BC2 e BC4: Paredes sem deslizamento (boundaries 2 e 4)
        bc2 = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), 
                         self.boundaries, 2)
        bc4 = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), 
                         self.boundaries, 4)

        self.bcs = [bc1, bc2, bc4]
        print("Condições de contorno configuradas")

    def setup_material_properties(self):
        """Define propriedades materiais variáveis espacialmente"""
        print("Configurando propriedades materiais...")
        # Expoentes de Corey
        # marker 0 = meio poroso (outer), marker 1 = vuggy (inner)
        self.no = {1: 1, 0: 2}
        self.nw = {1: 1, 0: 2}

        VVV = FunctionSpace(self.mesh, "DG", 0)

        noo = PiecewiseConstant(self.no, self.markers)
        self.noo_proj = project(noo, VVV)

        nww = PiecewiseConstant(self.nw, self.markers)
        self.nww_proj = project(nww, VVV)
        print("Propriedades materiais configuradas")

    def assemble_pressure_system(self):
        """Monta sistema para equação de pressão"""
        print("Montando sistema de pressão...")
        (u, p) = TrialFunctions(self.W)
        (v, q) = TestFunctions(self.W)

        dx = Measure("dx", domain=self.mesh, subdomain_data=self.markers)
        ds = Measure("ds", domain=self.mesh, subdomain_data=self.boundaries)
        n = FacetNormal(self.mesh)

        # Estabilização DG
        h = CellDiameter(self.mesh)
        h2 = ufl.Min(h("+"), h("-"))
        alpha = 35

        stab = (
            self.mu_w * (alpha / h2) * 
            inner(FlowEquations.tensor_jump(u, n), 
                  FlowEquations.tensor_jump(v, n)) * dS
            - self.mu_w * inner(avg(grad(u)), 
                                FlowEquations.tensor_jump(v, n)) * dS
            - self.mu_w * inner(avg(grad(v)), 
                                FlowEquations.tensor_jump(u, n)) * dS
        )

        Kinv = Constant(1 / self.k_matrix)
        f = Constant((0.0, 0.0))

        self.a_pressure = (
            FlowEquations.mu_brinkman(self.s0, self.mu_o, self.mu_w) * 
            inner(grad(u), grad(v)) * dx(1)
            + inner(v, FlowEquations.lambda_inv(self.s0, self.mu_w, self.mu_o,
                                                 self.no[0], self.nw[0]) * 
                   Kinv * u) * dx(0)
            - div(v) * p * dx
            + div(u) * q * dx
            + stab
        )

        self.L_pressure = (
            inner(f, v) * dx
            - self.pout_bc * dot(v, n) * ds(3)
        )
        print("Sistema de pressão montado")

    def assemble_saturation_system(self):
        """Monta sistema para equação de saturação"""
        print("Montando sistema de saturação...")
        s = TrialFunction(self.R)
        r = TestFunction(self.R)

        (u_, p_) = self.U.split()

        dx = Measure("dx", domain=self.mesh, subdomain_data=self.markers)
        ds = Measure("ds", domain=self.mesh, subdomain_data=self.boundaries)
        n = FacetNormal(self.mesh)

        dt_const = Constant(self.dt)
        sbar = Constant(1.0)  # Saturação de injeção

        # Fluxos upwind
        un = 0.5 * (inner(u_, n) + abs(inner(u_, n)))
        un_h = 0.5 * (inner(u_, n) - abs(inner(u_, n)))

        # Estabilização DG
        stabilisation = (
            dt_const("+") * inner(
                jump(r), 
                jump(un * FlowEquations.fractional_flow(
                    self.s0, self.mu_rel, self.noo_proj, self.nww_proj))
            ) * dS
        )

        # Forma completa da equação de saturação
        L3 = (
            self.phi * r * (s - self.s0) * dx(0)
            + r * (s - self.s0) * dx(1)
            - dt_const * inner(
                grad(r), 
                FlowEquations.fractional_flow(
                    self.s0, self.mu_rel, self.noo_proj, self.nww_proj) * u_
            ) * dx(0)
            - dt_const * inner(
                grad(r), 
                FlowEquations.fractional_flow_vugg(self.s0) * u_
            ) * dx(1)
            + dt_const * r * FlowEquations.fractional_flow(
                self.s0, self.mu_rel, self.no[0], self.nw[0]
            ) * un * ds
            + stabilisation
            + dt_const * r * un_h * sbar * ds(1)
        )

        self.a_saturation, self.L_saturation = lhs(L3), rhs(L3)
        print("Sistema de saturação montado")

    def solve_pressure(self):
        """Resolve equação de pressão"""
        solve(self.a_pressure == self.L_pressure, self.U, self.bcs)

    def solve_saturation(self):
        """Resolve equação de saturação (explícita)"""
        solve(self.a_saturation == self.L_saturation, self.S)
        self.s0.assign(self.S)

    # ==================== MÉTODOS DE PASSO DE TEMPO ADAPTATIVO ====================

    def compute_adaptive_timestep_classic(self):
        """
        Método CFL clássico (original) - mantido para comparação
        """
        if not self.adaptive_dt:
            return self.dt

        (u_, p_) = self.U.split()

        # Velocidade máxima no domínio
        u_array = u_.vector().get_local()
        u_magnitude = np.sqrt(u_array[::2]**2 + u_array[1::2]**2)
        u_max = np.max(u_magnitude)

        if u_max < 1e-15:
            dt_new = self.dt_max
        else:
            h_min = self.mesh.hmin()
            dt_new = self.CFL_max * h_min / u_max
            dt_new = max(min(dt_new, 1.5 * self.dt), 0.75 * self.dt)
            dt_new = max(min(dt_new, self.dt_max), self.dt_min)

        return dt_new

    def compute_adaptive_timestep_local(self):
        """
        Calcula dt baseado em CFL local por elemento, mais adequado para DG
        """
        if not self.adaptive_dt:
            return self.dt
        
        (u_, p_) = self.U.split()
        
        # Espaço DG0 para calcular por elemento
        DG0 = FunctionSpace(self.mesh, "DG", 0)
        
        # Projeta velocidade para DG0 (valor por elemento)
        u_proj = project(sqrt(inner(u_, u_)), DG0)
        
        # Diâmetro por elemento
        h = project(CellDiameter(self.mesh), DG0)
        
        # Mobilidade local (considera saturação e permeabilidade)
        lambda_total = project(
            (self.s0**self.nww_proj) / self.mu_w + 
            ((1.0 - self.s0)**self.noo_proj) / self.mu_o, 
            DG0
        )
        
        # CFL local considerando mobilidade
        u_array = u_proj.vector().get_local()
        h_array = h.vector().get_local()
        lambda_array = lambda_total.vector().get_local()
        
        # CFL efetivo por elemento
        cfl_local = np.where(
            lambda_array > 1e-12,
            u_array * np.sqrt(lambda_array) / h_array,
            0.0
        )
        
        # Remove valores muito pequenos
        cfl_local = cfl_local[cfl_local > 1e-12]
        
        if len(cfl_local) == 0:
            return self.dt_max
        
        # Usa percentil 95 ao invés do máximo (mais robusto)
        cfl_effective = np.percentile(cfl_local, 95)
        
        if cfl_effective < 1e-12:
            return self.dt_max
        
        # Calcula novo dt
        dt_new = self.CFL_max / cfl_effective
        
        # Limita variação (suavização)
        dt_new = max(min(dt_new, 1.2 * self.dt), 0.8 * self.dt)
        
        # Aplica limites globais
        dt_new = max(min(dt_new, self.dt_max), self.dt_min)
        
        return dt_new

    def compute_adaptive_timestep_saturation(self):
        """
        Controla dt baseado na taxa de variação da saturação
        """
        if not self.adaptive_dt or len(self.results['time']) < 2:
            return self.dt
        
        # Calcula variação máxima de saturação
        s_old = self.s0.vector().get_local()
        s_new = self.S.vector().get_local()
        
        ds_dt = np.abs(s_new - s_old) / self.dt
        
        # Remove valores muito pequenos
        ds_dt = ds_dt[ds_dt > 1e-10]
        
        if len(ds_dt) == 0:
            return self.dt_max
        
        # Usa percentil 90 para evitar outliers
        ds_dt_max = np.percentile(ds_dt, 90)
        
        # Critério: máxima variação de saturação por passo = 0.05
        max_ds_per_step = 0.05
        dt_saturation = max_ds_per_step / ds_dt_max
        
        # Combina com critério CFL
        dt_cfl = self.compute_adaptive_timestep_local()
        
        # Usa o mais restritivo
        dt_new = min(dt_saturation, dt_cfl)
        
        # Limita variação
        dt_new = max(min(dt_new, 1.3 * self.dt), 0.7 * self.dt)
        dt_new = max(min(dt_new, self.dt_max), self.dt_min)
        
        return dt_new

    def compute_adaptive_timestep_hybrid(self):
        """
        Método híbrido que combina múltiplos critérios
        """
        if not self.adaptive_dt:
            return self.dt
        
        dt_candidates = []
        
        # Critério 1: CFL baseado em Péclet local
        dt_peclet = self.compute_dt_peclet()
        if dt_peclet > 0:
            dt_candidates.append(dt_peclet)
        
        # Critério 2: Variação de saturação
        if len(self.results['time']) >= 2:
            dt_sat = self.compute_dt_saturation_variation()
            if dt_sat > 0:
                dt_candidates.append(dt_sat)
        
        # Critério 3: Estabilidade DG (diffusion)
        dt_dg = self.compute_dt_dg_stability()
        if dt_dg > 0:
            dt_candidates.append(dt_dg)
        
        # Critério 4: Baseado no residual
        if self.step > 10:  # Só após alguns passos
            dt_residual = self.compute_dt_residual()
            if dt_residual > 0:
                dt_candidates.append(dt_residual)
        
        if not dt_candidates:
            return self.dt_max
        
        # Usa o mais restritivo
        dt_new = min(dt_candidates)
        
        # Suavização temporal (evita oscilações)
        if hasattr(self, 'dt_history'):
            self.dt_history.append(dt_new)
            if len(self.dt_history) > 5:
                self.dt_history.pop(0)
            # Média móvel
            dt_new = np.mean(self.dt_history)
        else:
            self.dt_history = [dt_new]
        
        # Limita variação máxima por passo
        dt_new = max(min(dt_new, 1.1 * self.dt), 0.9 * self.dt)
        
        # Aplica limites globais
        dt_new = max(min(dt_new, self.dt_max), self.dt_min)
        
        return dt_new

    def compute_dt_peclet(self):
        """CFL baseado no número de Péclet local"""
        (u_, p_) = self.U.split()
        
        DG0 = FunctionSpace(self.mesh, "DG", 0)
        
        # Velocidade efetiva
        u_eff = project(sqrt(inner(u_, u_)), DG0)
        h = project(CellDiameter(self.mesh), DG0)
        
        # Difusão numérica do DG (proporcional a u*h)
        # Para DG upwind, a difusão numérica é ~0.5*u*h
        diffusion_num = 0.5 * u_eff * h
        
        u_array = u_eff.vector().get_local()
        h_array = h.vector().get_local()
        diff_array = diffusion_num.vector().get_local()
        
        # Péclet local = u*h/D_num
        peclet_local = np.where(
            diff_array > 1e-12,
            u_array * h_array / diff_array,
            0.0
        )
        
        # Remove zeros
        peclet_local = peclet_local[peclet_local > 1e-10]
        
        if len(peclet_local) == 0:
            return self.dt_max
        
        # Critério: Péclet < 2 para boa estabilidade
        peclet_max = np.percentile(peclet_local, 90)
        
        if peclet_max < 1e-10:
            return self.dt_max
        
        dt_new = 2.0 / peclet_max * self.CFL_max
        
        return dt_new

    def compute_dt_saturation_variation(self):
        """Baseado na taxa de variação da saturação"""
        s_old = self.s0.vector().get_local()
        s_new = self.S.vector().get_local()
        
        # Taxa de variação absoluta
        ds_abs = np.abs(s_new - s_old)
        
        # Remove variações muito pequenas
        ds_significant = ds_abs[ds_abs > 1e-6]
        
        if len(ds_significant) == 0:
            return self.dt_max
        
        # Máxima variação permitida por passo = 3%
        max_variation = 0.03
        ds_max = np.percentile(ds_significant, 95)
        
        # dt para manter variação sob controle
        dt_new = self.dt * max_variation / ds_max
        
        return dt_new

    def compute_dt_dg_stability(self):
        """Baseado na estabilidade específica do DG"""
        (u_, p_) = self.U.split()
        
        # Para DG, a estabilidade depende do parâmetro de penalização
        # alpha/h + |u|/h < C (C~10-20 para DG)
        
        DG0 = FunctionSpace(self.mesh, "DG", 0)
        h = project(CellDiameter(self.mesh), DG0)
        u_mag = project(sqrt(inner(u_, u_)), DG0)
        
        h_array = h.vector().get_local()
        u_array = u_mag.vector().get_local()
        
        # Parâmetro de penalização (usado na estabilização)
        alpha = 35  # Mesmo valor usado no código
        
        # Critério de estabilidade DG
        stability_param = alpha / h_array + u_array / h_array
        
        # Remove valores muito pequenos
        stability_param = stability_param[stability_param > 1e-10]
        
        if len(stability_param) == 0:
            return self.dt_max
        
        # Limite de estabilidade para DG
        stability_limit = 15.0
        
        stability_max = np.percentile(stability_param, 90)
        
        if stability_max < 1e-10:
            return self.dt_max
        
        dt_new = self.dt * stability_limit / stability_max
        
        return dt_new

    def compute_dt_residual(self):
        """Baseado na convergência do resíduo"""
        if len(self.results['Qw']) < 3:
            return self.dt_max
        
        # Analisa variação nas vazões (indicador de convergência)
        qw_recent = self.results['Qw'][-3:]
        qo_recent = self.results['Qo'][-3:]
        
        # Variação relativa
        qw_var = np.std(qw_recent) / (np.mean(qw_recent) + 1e-12)
        qo_var = np.std(qo_recent) / (np.mean(qo_recent) + 1e-12)
        
        total_var = qw_var + qo_var
        
        # Se variação alta, diminui dt
        # Se variação baixa, pode aumentar dt
        
        if total_var > 0.1:  # Muita oscilação
            return 0.8 * self.dt
        elif total_var < 0.01:  # Bem convergido
            return 1.3 * self.dt
        else:
            return self.dt

    def compute_adaptive_timestep(self):
        """
        Método principal que chama o método selecionado
        """
        methods = {
            "classic": self.compute_adaptive_timestep_classic,
            "local": self.compute_adaptive_timestep_local,
            "saturation": self.compute_adaptive_timestep_saturation,
            "hybrid": self.compute_adaptive_timestep_hybrid
        }
        
        if self.dt_method not in methods:
            print(f"Aviso: Método '{self.dt_method}' não reconhecido. Usando 'hybrid'.")
            self.dt_method = "hybrid"
        
        return methods[self.dt_method]()

    def update_timestep(self):
        """Atualiza o passo de tempo com método selecionado"""
        if self.adaptive_dt:
            dt_old = self.dt
            
            # Use o método selecionado
            self.dt = self.compute_adaptive_timestep()
            
            if abs(self.dt - dt_old) > 1e-10:
                print(f"  -> dt ajustado ({self.dt_method}): {dt_old:.6e} → {self.dt:.6e} s")
                # Precisa remontar o sistema de saturação com novo dt
                self.assemble_saturation_system()

    # ==================== CONTINUAÇÃO DO CÓDIGO ORIGINAL ====================

    def initialize_xdmf_files(self):
        """Inicializa arquivos XDMF para salvar campos"""
        viz_dir = self.output_dir / "visualization"
        viz_dir.mkdir(parents=True, exist_ok=True)

        self.u_file = XDMFFile(str(viz_dir / "velocity.xdmf"))
        self.p_file = XDMFFile(str(viz_dir / "pressure.xdmf"))
        self.s_file = XDMFFile(str(viz_dir / "saturation.xdmf"))

        self.u_file.parameters["flush_output"] = True
        self.p_file.parameters["flush_output"] = True
        self.s_file.parameters["flush_output"] = True

        self.u_file.parameters["rewrite_function_mesh"] = False
        self.p_file.parameters["rewrite_function_mesh"] = False
        self.s_file.parameters["rewrite_function_mesh"] = False

        print("Arquivos XDMF inicializados")

    def close_xdmf_files(self):
        """Fecha arquivos XDMF"""
        self.u_file.close()
        self.p_file.close()
        self.s_file.close()

    def save_fields(self):
        """Salva campos para visualização"""
        (u_, p_) = self.U.split(deepcopy=True)

        u_.rename("velocity", "velocity")
        p_.rename("pressure", "pressure")
        self.S.rename("saturation", "saturation")

        self.u_file.write(u_, self.t)
        self.p_file.write(p_, self.t)
        self.s_file.write(self.S, self.t)

        print(f"  -> Campos salvos em t = {self.t:.4f}s")

    def compute_diagnostics(self) -> Dict[str, float]:
        """Calcula diagnósticos da simulação"""
        (u_, p_) = self.U.split()

        ds = Measure("ds", domain=self.mesh, subdomain_data=self.boundaries)
        dx = Measure("dx", domain=self.mesh, subdomain_data=self.markers)
        n = FacetNormal(self.mesh)

        # Saturações médias
        Sin = float(assemble(self.S * ds(1))) / self.A_in
        Sout = float(assemble(self.S * ds(3))) / self.A_in
        Sdx = float(assemble(self.S * dx)) / self.Area

        # Vazões
        Q_total = -float(assemble(dot(u_, n) * ds(1)))

        # Vazão de água na saída
        Qw = float(assemble(
            FlowEquations.fractional_flow(
                self.s0, self.mu_rel, self.noo_proj, self.nww_proj
            ) * dot(u_, n) * ds(3)
        ))

        # Vazão de óleo
        Qo = Q_total - Qw

        # Pressão média na entrada
        pin = float(assemble(p_ * ds(1))) / self.A_in

        # Volume acumulado injetado
        if len(self.results['Vinj']) == 0:
            Vinj = Q_total * self.dt
        else:
            Vinj = self.results['Vinj'][-1] + Q_total * self.dt

        # Verifica balanço de massa
        uin = float(assemble(dot(u_, n) * ds(1)))
        uout = float(assemble(dot(u_, n) * ds(3)))
        mass_balance_error = abs(abs(uin) - abs(uout))

        print(f"  Método dt: {self.dt_method} | dt atual={self.dt:.6e} s")
        print(f"  Sin={Sin:.4f}, Sout={Sout:.4f}, Sdx={Sdx:.4f}")
        print(f"  Qo={Qo:.6e}, Qw={Qw:.6e}, pin={pin:.2f} Pa")
        print(f"  Balanço de massa: erro={mass_balance_error:.6e}")

        return {
            'time': self.t,
            'dt': self.dt,
            'Qo': Qo,
            'Qw': Qw,
            'pin': pin,
            'pout': self.pout_bc,
            'Vinj': Vinj,
            'Sin': Sin,
            'Sout': Sout,
            'Sdx': Sdx
        }

    def check_convergence(self) -> bool:
        """Verifica critérios de parada"""
        if len(self.results['Sout']) < 2:
            return False

        Sout = self.results['Sout'][-1]

        # Critério: se saturação de saída > 0.3 e razão óleo/água < 0.05
        if Sout > 0.3:
            Qo = self.results['Qo'][-1]
            Qw = self.results['Qw'][-1]

            if Qw > 1e-12:  # Evita divisão por zero
                ratio = Qo / Qw
                print(f"  Razão Qo/Qw = {ratio:.6f}")

                if ratio < 0.05:
                    return True

        return False

    def save_checkpoint(self, checkpoint_name: str = "checkpoint"):
        """
        Salva estado completo da simulação para restart

        Args:
            checkpoint_name: Nome do checkpoint (sem extensão)
        """
        import json

        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_file = checkpoint_dir / f"{checkpoint_name}.h5"
        metadata_file = checkpoint_dir / f"{checkpoint_name}_metadata.json"
        results_file = checkpoint_dir / f"{checkpoint_name}_results.json"

        print(f"  -> Salvando checkpoint: {checkpoint_name}")

        # Salva campos FEniCS
        hdf = HDF5File(self.mesh.mpi_comm(), str(checkpoint_file), "w")
        hdf.write(self.U, "velocity_pressure")
        hdf.write(self.S, "saturation")
        hdf.write(self.s0, "saturation_old")
        hdf.close()

        # Salva metadados da simulação
        metadata = {
            'time': float(self.t),
            'step': int(self.step),
            'dt': float(self.dt),
            'dt_initial': float(self.dt_initial),
            'adaptive_dt': self.adaptive_dt,
            'dt_method': self.dt_method,
            'CFL_max': float(self.CFL_max),
            'dt_min': float(self.dt_min),
            'dt_max': float(self.dt_max),
            'mu_w': float(self.mu_w),
            'mu_o': float(self.mu_o),
            'phi': float(self.phi),
            'k_matrix': float(self.k_matrix),
            'inlet_velocity': float(self.inlet_velocity),
            'pin_value': float(self.pin_value),
            'pout_value': float(self.pout_value)
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Salva histórico de resultados
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"     Checkpoint salvo com sucesso!")

    def load_checkpoint(self, checkpoint_name: str = "checkpoint"):
        """
        Carrega estado da simulação de um checkpoint

        Args:
            checkpoint_name: Nome do checkpoint (sem extensão)

        Returns:
            True se carregou com sucesso, False caso contrário
        """
        import json

        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_file = checkpoint_dir / f"{checkpoint_name}.h5"
        metadata_file = checkpoint_dir / f"{checkpoint_name}_metadata.json"
        results_file = checkpoint_dir / f"{checkpoint_name}_results.json"

        # Verifica se arquivos existem
        if not checkpoint_file.exists():
            print(f"Checkpoint não encontrado: {checkpoint_file}")
            return False

        print(f"Carregando checkpoint: {checkpoint_name}")

        try:
            # Carrega campos FEniCS
            hdf = HDF5File(self.mesh.mpi_comm(), str(checkpoint_file), "r")
            hdf.read(self.U, "velocity_pressure")
            hdf.read(self.S, "saturation")
            hdf.read(self.s0, "saturation_old")
            hdf.close()

            # Carrega metadados
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            self.t = metadata['time']
            self.step = metadata['step']
            self.dt = metadata['dt']
            self.dt_initial = metadata['dt_initial']
            
            # Carrega método de dt se disponível
            if 'dt_method' in metadata:
                self.dt_method = metadata['dt_method']

            # Carrega histórico de resultados
            with open(results_file, 'r') as f:
                self.results = json.load(f)

            print(f"  Checkpoint carregado com sucesso!")
            print(f"  Reiniciando do passo {self.step}, t = {self.t:.4f} s")
            print(f"  Método de dt: {self.dt_method}")

            return True

        except Exception as e:
            print(f"Erro ao carregar checkpoint: {e}")
            return False

    def run(self, T: float, impes_steps: int = 1, 
            save_interval: int = 50, max_steps: int = int(1e6),
            restart_from_checkpoint: str = None):
        """
        Executa simulação

        Args:
            T: Tempo final de simulação [s]
            impes_steps: Frequência de atualização da pressão
            save_interval: Intervalo para salvar resultados
            max_steps: Número máximo de passos
            restart_from_checkpoint: Nome do checkpoint para reiniciar (None = nova simulação)
        """
        print("\n" + "="*60)
        print("INICIANDO SIMULAÇÃO BRINKMAN-IMPES")
        print("="*60)

        # Setup inicial
        self.load_mesh()
        self.setup_function_spaces()
        self.setup_boundary_conditions()
        self.setup_material_properties()
        self.assemble_pressure_system()
        self.assemble_saturation_system()

        # Tenta carregar checkpoint se especificado
        if restart_from_checkpoint is not None:
            if self.load_checkpoint(restart_from_checkpoint):
                print("\n*** REINICIANDO DE CHECKPOINT ***\n")
            else:
                print("\n*** CHECKPOINT NÃO ENCONTRADO - INICIANDO NOVA SIMULAÇÃO ***\n")

        # Inicializa arquivos apenas se for nova simulação
        if restart_from_checkpoint is None or self.step == 0:
            self.writer.initialize_file()
            self.initialize_xdmf_files()
        else:
            # Reabre arquivos XDMF para append
            self.initialize_xdmf_files()

        # Calcula áreas
        ds = Measure("ds", domain=self.mesh, subdomain_data=self.boundaries)
        dx = Measure("dx", domain=self.mesh, subdomain_data=self.markers)
        self.A_in = float(assemble(Constant(1.0) * ds(1)))
        self.Area = float(assemble(Constant(1.0) * dx))

        print(f"\nÁrea de entrada: {self.A_in:.6e} m²")
        print(f"Área total: {self.Area:.6e} m²")
        print(f"\nParâmetros da simulação:")
        print(f"  - Tempo final: {T} s")
        print(f"  - Passo de tempo inicial: {self.dt} s")
        if self.adaptive_dt:
            print(f"  - Passo de tempo ADAPTATIVO ativado")
            print(f"    * Método: {self.dt_method}")
            print(f"    * CFL_max: {self.CFL_max}")
            print(f"    * dt_min: {self.dt_min:.6e} s")
            print(f"    * dt_max: {self.dt_max:.6e} s")
        else:
            print(f"  - Passo de tempo FIXO")
        print(f"  - Velocidade de entrada: {self.inlet_velocity} m/s")
        print(f"  - IMPES steps: {impes_steps}")
        print(f"  - Save interval: {save_interval}")
        if self.checkpoint_interval > 0:
            print(f"  - Checkpoint interval: {self.checkpoint_interval} passos")
        print("\n" + "="*60 + "\n")

        # Loop temporal
        while self.step < max_steps and self.t < T:
            start = time.time()

            # Resolve pressão (IMPES) - no primeiro passo e periodicamente
            if self.step == 0 or (self.step % impes_steps == 0 and self.step > 0):
                self.solve_pressure()
                if self.step > 0:
                    print(f"  [Pressão atualizada no passo {self.step}]")

            # Resolve saturação
            self.solve_saturation()

            # Atualiza passo de tempo se adaptativo
            if self.adaptive_dt and self.step > 0:
                self.update_timestep()

            # Atualiza tempo
            self.t += self.dt

            # Diagnósticos
            timestep_data = self.compute_diagnostics()

            # Salva dados
            self.writer.append_timestep(timestep_data)
            for key, value in timestep_data.items():
                self.results[key].append(value)

            # Salva campos
            if self.step % save_interval == 0:
                self.save_fields()

            # Salva checkpoint periodicamente
            if self.checkpoint_interval > 0 and self.step % self.checkpoint_interval == 0 and self.step > 0:
                self.save_checkpoint(f"checkpoint_step_{self.step}")

            # Verifica convergência
            if self.check_convergence():
                print(f"\n*** Convergência atingida no passo {self.step} ***\n")
                # Salva checkpoint final antes de sair
                if self.checkpoint_interval > 0:
                    self.save_checkpoint("checkpoint_final")
                break

            self.step += 1

            elapsed = time.time() - start

            # Informação sobre o passo de tempo
            if self.adaptive_dt:
                print(f"Passo {self.step} concluído: t = {self.t:.4f}s | dt = {self.dt:.6e}s | tempo: {elapsed:.3f}s\n")
            else:
                print(f"Passo {self.step} concluído: t = {self.t:.4f}s | dt fixo = {self.dt:.6e}s | tempo: {elapsed:.3f}s\n")

        # Salva checkpoint final
        if self.checkpoint_interval > 0:
            self.save_checkpoint("checkpoint_final")

        # Finaliza
        self.close_xdmf_files()
        self.writer.write_summary(self.results)

        print("\n" + "="*60)
        print("SIMULAÇÃO CONCLUÍDA COM SUCESSO!")
        print("="*60)
        print(f"Total de passos: {self.step}")
        print(f"Tempo final: {self.t:.2f} s")
        print(f"Método de dt usado: {self.dt_method}")
        print(f"Resultados salvos em: {self.output_dir}")
        print("="*60 + "\n")

# ==================== EXEMPLOS DE USO ====================

if __name__ == "__main__":

    # ========== EXEMPLO 1: Método híbrido (Recomendado) ==========
    print("\n>>> EXEMPLO 1: Método híbrido (RECOMENDADO) > EXEMPLO 2: Método baseado em saturação > EXEMPLO 3: Método CFL local > EXEMPLO 4: Para comparar métodos <<<\n")

    # Executa mesmo caso com diferentes métodos para comparação
    
    methods_to_test = ["classic", "local", "saturation", "hybrid"]
    
    for method in methods_to_test:
        print(f"\n--- Testando método: {method} ---")
        
        solver_test = BrinkmanIMPESSolver(
            mesh_dir="/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/Arapua24/mesh",
            output_dir=f"/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/Arapua24/results_{method}",
            mu_w=1e-3,
            mu_o=1e-3,
            perm_darcy=100,
            dt=0.1,
            adaptive_dt=True,
            CFL_max=0.5,
            dt_min=1e-5,
            dt_max=20.0,
            dt_method=method,
            checkpoint_interval=0
        )
        
        # solver_test.run(T=20.0, impes_steps=50, save_interval=20)
        
        print(f"--- Método {method} finalizado ---\n")

    print("\n*** TODAS AS SIMULAÇÕES CONCLUÍDAS ***")
    print("Compare os resultados nos diretórios:")
    print("  - results_classic/")
    print("  - results_local/")
    print("  - results_saturation/")
    print("  - results_hybrid/")
