"""
Biblioteca para simulação de fluxo bifásico usando método Brinkman-IMPES
Versão Melhorada - Com estratégia robusta de passo de tempo adaptativo por fases
e CONTROLE ADAPTATIVO DE SATURAÇÃO POR FASE
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
                   "Vinj", "Sin", "Sout", "Sdx", "phase", "convergence_failures", "saturation_limit"]

        with open(self.results_file, 'w') as f:
            f.write(','.join(headers) + '\n')

        print(f"Arquivo inicializado: {self.results_file}")

    def append_timestep(self, data: Dict[str, float]):
        """Adiciona dados de um passo de tempo"""
        keys = ["time", "dt", "Qo", "Qw", "pin", "pout", 
                "Vinj", "Sin", "Sout", "Sdx", "phase", "convergence_failures", "saturation_limit"]

        with open(self.results_file, 'a') as f:
            row = ','.join(str(data[key]) for key in keys)
            f.write(row + '\n')

    def write_summary(self, data_arrays: Dict[str, list]):
        """Escreve resumo completo da simulação"""
        if not data_arrays or len(data_arrays.get('time', [])) == 0:
            print("Aviso: Nenhum dado para gravar no summary")
            return

        keys = ["time", "dt", "Qo", "Qw", "pin", "pout", 
                "Vinj", "Sin", "Sout", "Sdx", "phase", "convergence_failures", "saturation_limit"]

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
    equação de Brinkman para meios vuggy e estratégia robusta
    de passo de tempo adaptativo por fases com CONTROLE ADAPTATIVO
    DE SATURAÇÃO POR FASE
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
                 dt_method: str = "robust_phases",
                 saturation_limits: Dict[int, float] = None):
        """
        Inicializa o solver com estratégia robusta de passo de tempo e
        CONTROLE ADAPTATIVO DE SATURAÇÃO POR FASE

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
            CFL_max: Número de CFL máximo para dt adaptativo (usado na Fase 1)
            dt_min: Passo de tempo mínimo [s]
            dt_max: Passo de tempo máximo [s]
            checkpoint_interval: Salva checkpoint a cada N passos (0 = desativado)
            dt_method: Método de cálculo de dt ("robust_phases" é o novo método)
            saturation_limits: Limites de saturação por fase {fase: max_change}
                              Se None, usa padrão: {1: 0.05, 2: 0.1, 3: 0.15}
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

        # Parâmetros numéricos com validação
        self.dt = max(min(dt, dt_max), dt_min)  # Garante que dt inicial está nos limites
        self.dt_initial = self.dt
        self.adaptive_dt = adaptive_dt
        self.CFL_max = CFL_max
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_method = dt_method
        self.checkpoint_interval = checkpoint_interval
        self.t = 0.0
        self.step = 0

        # Validação de limites
        if self.dt_min >= self.dt_max:
            raise ValueError(f"dt_min ({dt_min:.2e}) deve ser menor que dt_max ({dt_max:.2e})")

        # ========== NOVA ESTRATÉGIA ROBUSTA POR FASES ==========
        
        # Controle de fases
        self.phase = 1  # Fase atual (1: início, 2: desenvolvimento, 3: regime)
        self.consecutive_convergence = 0
        self.consecutive_failures = 0
        self.last_dt_values = []  # Histórico dos últimos dt

        # Parâmetros por fase
        self.phase_params = {
            1: {'cfl_max': 0.5, 'max_steps': 20, 'dt_variation': 1.1, 'name': 'início'},
            2: {'cfl_max': 0.8, 'max_steps': 100, 'dt_variation': 1.2, 'name': 'desenvolvimento'},
            3: {'cfl_max': 1.0, 'max_steps': float('inf'), 'dt_variation': 1.2, 'name': 'regime'}
        }

        # ========== CONTROLE ADAPTATIVO DE SATURAÇÃO POR FASE ==========
        # NOVO: Limites de saturação específicos para cada fase
        if saturation_limits is None:
            self.saturation_control_by_phase = {
                1: 0.05,  # Fase 1 (início): muito conservativo - 5%
                2: 0.1,   # Fase 2 (desenvolvimento): padrão - 10%
                3: 0.15   # Fase 3 (regime): mais relaxado - 15%
            }
        else:
            self.saturation_control_by_phase = saturation_limits.copy()

        # Validação dos limites de saturação
        for phase, limit in self.saturation_control_by_phase.items():
            if not (0.01 <= limit <= 1.0):
                raise ValueError(f"Limite de saturação para fase {phase} ({limit}) deve estar entre 0.01 e 1.0")

        # Threshold de convergência
        self.convergence_threshold = 15

        # Componentes FEniCS (inicializados em setup)
        self.mesh = None
        self.W = None
        self.R = None
        self.U = None
        self.S = None

        # Escritor de dados
        self.writer = DataWriter(self.output_dir / "data")

        # Histórico de resultados (expandido com limite de saturação)
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
            'Sdx': [],
            'phase': [],
            'convergence_failures': [],
            'saturation_limit': []  # NOVO: registra limite de saturação usado
        }

        print(f"BrinkmanIMPESSolver inicializado com CONTROLE ADAPTATIVO DE SATURAÇÃO:")
        print(f"  dt_min = {self.dt_min:.2e} s")
        print(f"  dt_max = {self.dt_max:.2e} s") 
        print(f"  dt_inicial = {self.dt:.2e} s")
        print(f"  Limites de saturação por fase:")
        for phase, limit in self.saturation_control_by_phase.items():
            phase_name = self.phase_params[phase]['name']
            print(f"    Fase {phase} ({phase_name}): {limit*100:.1f}%")

    def get_current_saturation_limit(self) -> float:
        """
        Retorna o limite de variação de saturação para a fase atual
        
        Returns:
            Limite máximo de variação de saturação (0-1)
        """
        return self.saturation_control_by_phase[self.phase]

    def _validate_dt(self, dt_value: float, context: str = "") -> float:
        """
        Valida e corrige um valor de dt, garantindo que está nos limites
        
        Args:
            dt_value: Valor de dt a ser validado
            context: Contexto para debug (nome do método que chamou)
            
        Returns:
            dt corrigido e dentro dos limites
        """
        if not np.isfinite(dt_value) or dt_value <= 0:
            print(f"  Aviso {context}: dt inválido ({dt_value}), usando dt_min")
            return self.dt_min
            
        dt_corrected = np.clip(dt_value, self.dt_min, self.dt_max)
        
        if abs(dt_corrected - dt_value) > 1e-15:
            print(f"  Aviso {context}: dt ajustado de {dt_value:.2e} para {dt_corrected:.2e}")
            
        return dt_corrected

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

        # Valida dt antes de usar
        validated_dt = self._validate_dt(self.dt, "assemble_saturation")
        dt_const = Constant(validated_dt)
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
        """Resolve equação de pressão com tratamento de erro"""
        try:
            solve(self.a_pressure == self.L_pressure, self.U, self.bcs)
            return True
        except Exception as e:
            print(f"    Erro na equação de pressão: {e}")
            return False

    def solve_saturation(self):
        """Resolve equação de saturação (explícita) com tratamento de erro"""
        try:
            solve(self.a_saturation == self.L_saturation, self.S)
            
            # Limita saturação entre 0 e 1
            s_array = self.S.vector().get_local()
            s_array = np.clip(s_array, 0.0, 1.0)
            self.S.vector().set_local(s_array)
            self.S.vector().apply('insert')
            
            self.s0.assign(self.S)
            return True
        except Exception as e:
            print(f"    Erro na equação de saturação: {e}")
            return False

    # ==================== NOVA ESTRATÉGIA ROBUSTA POR FASES COM CONTROLE ADAPTATIVO DE SATURAÇÃO ====================

    def compute_adaptive_timestep_local(self):
        """
        Método CFL local com proteções explícitas de limite
        """
        if not self.adaptive_dt:
            return self._validate_dt(self.dt, "local_fixed")

        try:
            (u_, p_) = self.U.split()

            # Projeta velocidade para DG0
            DG0 = FunctionSpace(self.mesh, "DG", 0)
            u_magnitude = project(sqrt(inner(u_, u_)), DG0)
            h_cell = project(CellDiameter(self.mesh), DG0)

            u_array = u_magnitude.vector().get_local()
            h_array = h_cell.vector().get_local()

            # Evita divisão por zero
            h_array = np.where(h_array > 1e-12, h_array, 1e-12)

            # CFL local
            cfl_array = u_array / h_array
            cfl_array = cfl_array[cfl_array > 1e-12]

            if len(cfl_array) == 0:
                return self._validate_dt(self.dt_max, "local_no_velocity")

            # Usa percentil 95 para robustez
            cfl_max = np.percentile(cfl_array, 95)

            if cfl_max < 1e-12:
                return self._validate_dt(self.dt_max, "local_low_cfl")

            # CFL_max baseado na fase atual
            cfl_target = self.phase_params[self.phase]['cfl_max']
            dt_new = cfl_target / cfl_max

            return self._validate_dt(dt_new, "local_computed")

        except Exception as e:
            print(f"    Erro no cálculo CFL local: {e}")
            return self._validate_dt(min(self.dt, self.dt_max), "local_error")

    def compute_saturation_check(self):
        """
        Verificação de saturação com CONTROLE ADAPTATIVO POR FASE
        """
        try:
            if len(self.results['time']) < 1:
                return self._validate_dt(self.dt_max, "saturation_initial")

            # ⭐ NOVO: Usa limite da fase atual
            current_limit = self.get_current_saturation_limit()

            # Variação máxima de saturação
            s_array = self.S.vector().get_local()
            s_old_array = self.s0.vector().get_local()

            max_change = np.max(np.abs(s_array - s_old_array))

            # Se variação é muito grande, limita dt
            if max_change > current_limit:
                # Reduz dt proporcionalmente
                factor = current_limit / max_change
                dt_limited = self.dt * factor * 0.8  # margem de segurança
                
                print(f"    Limite de saturação (Fase {self.phase}): {current_limit*100:.1f}% | Variação atual: {max_change*100:.1f}%")
                
                return self._validate_dt(dt_limited, f"saturation_limited_phase_{self.phase}")

            return self._validate_dt(self.dt_max, "saturation_ok")

        except Exception as e:
            print(f"    Erro na verificação de saturação: {e}")
            return self._validate_dt(self.dt, "saturation_error")

    def compute_adaptive_timestep_robust_phases(self):
        """
        Nova estratégia robusta com múltiplas proteções de limite e
        CONTROLE ADAPTATIVO DE SATURAÇÃO POR FASE
        GARANTIA: dt_min ≤ dt_new ≤ dt_max SEMPRE
        """
        if not self.adaptive_dt:
            return self._validate_dt(self.dt, "robust_fixed")

        try:
            # Critério principal: CFL local (já protegido internamente)
            dt_cfl = self.compute_adaptive_timestep_local()

            # ⭐ NOVO: Critério de segurança com limite adaptativo por fase
            dt_sat_check = self.compute_saturation_check()

            # Usa o mais restritivo
            dt_new = min(dt_cfl, dt_sat_check)

            # Aplicar limitação de variação baseada na fase
            max_variation = self.phase_params[self.phase]['dt_variation']
            dt_upper = min(max_variation * self.dt, self.dt_max)  # Nunca excede dt_max
            dt_lower = max(self.dt / max_variation, self.dt_min)  # Nunca fica abaixo de dt_min
            
            dt_new = np.clip(dt_new, dt_lower, dt_upper)

            # Proteção final absoluta
            dt_new = self._validate_dt(dt_new, "robust_final")

            # Verificação paranóica (debug)
            assert self.dt_min <= dt_new <= self.dt_max, \
                f"ERRO CRÍTICO: dt fora dos limites: {dt_new:.2e} não está em [{self.dt_min:.2e}, {self.dt_max:.2e}]"

            return dt_new

        except Exception as e:
            print(f"    ERRO no cálculo robustos por fases: {e}")
            print(f"    Retornando dt seguro = {min(self.dt, self.dt_max):.2e}")
            return self._validate_dt(min(self.dt, self.dt_max), "robust_error_recovery")

    def check_phase_transition(self):
        """Verifica se deve mudar de fase com logs de limites de saturação"""
        if self.phase == 1:
            # Fase 1 → 2: após passos bem-sucedidos
            if (self.step >= self.phase_params[1]['max_steps'] and 
                self.consecutive_convergence >= 10):
                old_limit = self.get_current_saturation_limit()
                self.phase = 2
                new_limit = self.get_current_saturation_limit()
                print(f"  → Transição para Fase 2 ({self.phase_params[2]['name']})")
                print(f"    CFL_max: {self.phase_params[1]['cfl_max']} → {self.phase_params[2]['cfl_max']}")
                print(f"    Limite saturação: {old_limit*100:.1f}% → {new_limit*100:.1f}%")

        elif self.phase == 2:
            # Fase 2 → 3: quando saturação está estabelecida
            if (self.step >= self.phase_params[2]['max_steps'] or
                (len(self.results['Sin']) >= 10 and
                 np.std(self.results['Sin'][-10:]) < 0.01)):
                old_limit = self.get_current_saturation_limit()
                self.phase = 3
                new_limit = self.get_current_saturation_limit()
                print(f"  → Transição para Fase 3 ({self.phase_params[3]['name']})")
                print(f"    CFL_max: {self.phase_params[2]['cfl_max']} → {self.phase_params[3]['cfl_max']}")
                print(f"    Limite saturação: {old_limit*100:.1f}% → {new_limit*100:.1f}%")

    def handle_convergence_failure(self):
        """Gerencia falhas de convergência com proteções"""
        self.consecutive_failures += 1
        self.consecutive_convergence = 0

        if self.consecutive_failures == 1:
            # Primeira falha: reduz dt moderadamente
            dt_new = self.dt * 0.7
            self.dt = self._validate_dt(dt_new, "failure_moderate")
            print(f"    Falha de convergência. Reduzindo dt para {self.dt:.2e}")

        elif self.consecutive_failures <= 3:
            # Falhas múltiplas: redução mais agressiva
            dt_new = self.dt * 0.5
            self.dt = self._validate_dt(dt_new, "failure_aggressive")
            print(f"    Falha {self.consecutive_failures}. dt = {self.dt:.2e}")

        else:
            # Muitas falhas: volta para dt_min e Fase 1
            self.dt = self.dt_min
            old_phase = self.phase
            self.phase = 1
            self.consecutive_failures = 0
            old_limit = self.saturation_control_by_phase[old_phase]
            new_limit = self.get_current_saturation_limit()
            print(f"    Reset para Fase 1. dt = {self.dt:.2e}")
            print(f"    Limite saturação: {old_limit*100:.1f}% → {new_limit*100:.1f}%")

        # Garante que está nos limites (redundante, mas seguro)
        self.dt = self._validate_dt(self.dt, "failure_final")

        # Remonta sistema de saturação com novo dt
        self.assemble_saturation_system()

    def compute_adaptive_timestep(self):
        """
        Método principal que chama o método selecionado com proteções
        """
        try:
            if self.dt_method == "robust_phases":
                return self.compute_adaptive_timestep_robust_phases()
            else:
                # Mantém métodos originais para compatibilidade
                methods = {
                    "classic": self.compute_adaptive_timestep_classic,
                    "local": self.compute_adaptive_timestep_local,
                }
                
                if self.dt_method in methods:
                    dt_computed = methods[self.dt_method]()
                    return self._validate_dt(dt_computed, f"method_{self.dt_method}")
                else:
                    print(f"Aviso: Método '{self.dt_method}' não reconhecido. Usando 'robust_phases'.")
                    return self.compute_adaptive_timestep_robust_phases()
                    
        except Exception as e:
            print(f"ERRO no compute_adaptive_timestep: {e}")
            return self._validate_dt(self.dt, "main_error_recovery")

    def update_timestep(self):
        """Atualiza passo de tempo com nova estratégia robusta e proteções"""
        if not self.adaptive_dt:
            return

        dt_old = self.dt

        try:
            # Calcula novo dt (já protegido internamente)
            self.dt = self.compute_adaptive_timestep()

            # Verificação final paranóica
            self.dt = self._validate_dt(self.dt, "update_final")

            # Atualiza histórico
            self.last_dt_values.append(self.dt)
            if len(self.last_dt_values) > 5:
                self.last_dt_values.pop(0)

            # Verifica transição de fase
            self.check_phase_transition()

            if abs(self.dt - dt_old) > 1e-12:
                current_limit = self.get_current_saturation_limit()
                print(f"  -> dt ajustado (Fase {self.phase} - {self.phase_params[self.phase]['name']} | Sat: {current_limit*100:.1f}%): {dt_old:.6e} → {self.dt:.6e} s")
                # Precisa remontar o sistema de saturação com novo dt
                self.assemble_saturation_system()

        except Exception as e:
            print(f"ERRO CRÍTICO em update_timestep: {e}")
            print(f"Mantendo dt anterior: {dt_old:.2e}")
            self.dt = self._validate_dt(dt_old, "update_error_recovery")

    # ==================== MÉTODOS ORIGINAIS MANTIDOS COM PROTEÇÕES ====================

    def compute_adaptive_timestep_classic(self):
        """Método CFL clássico com proteções de limite"""
        if not self.adaptive_dt:
            return self._validate_dt(self.dt, "classic_fixed")

        try:
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

            return self._validate_dt(dt_new, "classic_computed")

        except Exception as e:
            print(f"    Erro no método clássico: {e}")
            return self._validate_dt(self.dt, "classic_error")

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
        """Calcula diagnósticos da simulação (expandido com limite de saturação)"""
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

        # Limite de saturação atual
        saturation_limit = self.get_current_saturation_limit()

        # Verifica balanço de massa
        uin = float(assemble(dot(u_, n) * ds(1)))
        uout = float(assemble(dot(u_, n) * ds(3)))
        mass_balance_error = abs(abs(uin) - abs(uout))

        print(f"  Fase {self.phase} ({self.phase_params[self.phase]['name']}) | Sat.Limit: {saturation_limit*100:.1f}% | dt={self.dt:.6e} s")
        print(f"  Convergência: {self.consecutive_convergence} | Falhas: {self.consecutive_failures}")
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
            'Sdx': Sdx,
            'phase': self.phase,
            'convergence_failures': self.consecutive_failures,
            'saturation_limit': saturation_limit  # NOVO
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
        Salva estado completo da simulação para restart (expandido com controle de saturação)
        """
        import json

        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_file = checkpoint_dir / f"{checkpoint_name}.h5"
        metadata_file = checkpoint_dir / f"{checkpoint_name}_metadata.json"
        results_file = checkpoint_dir / f"{checkpoint_name}_results.json"

        print(f"  -> Salvando checkpoint: {checkpoint_name}")

        try:
            # Salva campos FEniCS
            hdf = HDF5File(self.mesh.mpi_comm(), str(checkpoint_file), "w")
            hdf.write(self.U, "velocity_pressure")
            hdf.write(self.S, "saturation")
            hdf.write(self.s0, "saturation_old")
            hdf.close()

            # Salva metadados da simulação (expandido com controle de saturação)
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
                'pout_value': float(self.pout_value),
                # Campos da estratégia robusta
                'phase': int(self.phase),
                'consecutive_convergence': int(self.consecutive_convergence),
                'consecutive_failures': int(self.consecutive_failures),
                # NOVO: Controle de saturação por fase
                'saturation_control_by_phase': self.saturation_control_by_phase
            }

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Salva histórico de resultados
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)

            print(f"     Checkpoint salvo com sucesso!")

        except Exception as e:
            print(f"     ERRO ao salvar checkpoint: {e}")

    def load_checkpoint(self, checkpoint_name: str = "checkpoint"):
        """
        Carrega estado da simulação de um checkpoint (expandido com controle de saturação)
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
            
            # Valida dt antes de usar
            dt_loaded = metadata['dt']
            self.dt = self._validate_dt(dt_loaded, "checkpoint_load")
            
            self.dt_initial = metadata['dt_initial']

            # Carrega método de dt se disponível
            if 'dt_method' in metadata:
                self.dt_method = metadata['dt_method']

            # Carrega campos da estratégia robusta
            if 'phase' in metadata:
                self.phase = metadata['phase']
                self.consecutive_convergence = metadata.get('consecutive_convergence', 0)
                self.consecutive_failures = metadata.get('consecutive_failures', 0)

            # NOVO: Carrega controle de saturação por fase se disponível
            if 'saturation_control_by_phase' in metadata:
                # Converte chaves string para int (JSON não suporta chaves int)
                sat_control = metadata['saturation_control_by_phase']
                if isinstance(list(sat_control.keys())[0], str):
                    self.saturation_control_by_phase = {int(k): v for k, v in sat_control.items()}
                else:
                    self.saturation_control_by_phase = sat_control

            # Carrega histórico de resultados
            with open(results_file, 'r') as f:
                self.results = json.load(f)

            print(f"  Checkpoint carregado com sucesso!")
            print(f"  Reiniciando do passo {self.step}, t = {self.t:.4f} s")
            print(f"  Método de dt: {self.dt_method}")
            print(f"  Fase atual: {self.phase} ({self.phase_params[self.phase]['name']})")
            print(f"  Limite saturação atual: {self.get_current_saturation_limit()*100:.1f}%")
            print(f"  dt carregado: {self.dt:.6e} s [validado]")

            return True

        except Exception as e:
            print(f"Erro ao carregar checkpoint: {e}")
            return False

    def run(self, T: float, impes_steps: int = 1, 
            save_interval: int = 50, max_steps: int = int(1e6),
            restart_from_checkpoint: str = None):
        """
        Executa simulação com estratégia robusta por fases e CONTROLE ADAPTATIVO DE SATURAÇÃO
        """
        print("\n" + "="*90)
        print("INICIANDO SIMULAÇÃO BRINKMAN-IMPES COM CONTROLE ADAPTATIVO DE SATURAÇÃO POR FASE")
        print("="*90)

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
        print(f"  - Passo de tempo inicial: {self.dt:.6e} s")
        print(f"  - Método de dt: {self.dt_method}")
        print(f"  - CONTROLE ADAPTATIVO DE SATURAÇÃO POR FASE ATIVADO:")
        for phase, limit in self.saturation_control_by_phase.items():
            phase_name = self.phase_params[phase]['name']
            print(f"    * Fase {phase} ({phase_name}): max Δs = {limit*100:.1f}%")
        if self.adaptive_dt:
            print(f"  - ESTRATÉGIA ROBUSTA POR FASES ativada")
            print(f"    * Fase inicial: {self.phase} ({self.phase_params[self.phase]['name']})")
            print(f"    * CFL_max inicial: {self.phase_params[self.phase]['cfl_max']}")
            print(f"    * LIMITES RIGOROSOS: dt ∈ [{self.dt_min:.2e}, {self.dt_max:.2e}] s")
        else:
            print(f"  - Passo de tempo FIXO")
        print(f"  - Velocidade de entrada: {self.inlet_velocity} m/s")
        print(f"  - IMPES steps: {impes_steps}")
        print(f"  - Save interval: {save_interval}")
        if self.checkpoint_interval > 0:
            print(f"  - Checkpoint interval: {self.checkpoint_interval} passos")
        print("\n" + "="*90 + "\n")

        # Loop temporal com estratégia robusta e controle de saturação
        while self.step < max_steps and self.t < T:
            start = time.time()

            print(f"\n=== Passo {self.step + 1}, t = {self.t:.4e} ===")

            step_success = True

            # Resolve pressão (IMPES) - no primeiro passo e periodicamente
            if self.step == 0 or (self.step % impes_steps == 0 and self.step > 0):
                print("  Resolvendo campo de velocidade...")
                pressure_success = self.solve_pressure()
                if not pressure_success:
                    step_success = False
                    print("  ✗ Falha na equação de pressão")
                elif self.step > 0:
                    print(f"  [Pressão atualizada no passo {self.step + 1}]")

            # Resolve saturação
            if step_success:
                print("  Resolvendo transporte...")
                saturation_success = self.solve_saturation()
                if not saturation_success:
                    step_success = False
                    print("  ✗ Falha na equação de saturação")

            # Verifica sucesso do passo
            if step_success:
                # Sucesso: atualiza passo de tempo se adaptativo
                if self.adaptive_dt and self.step > 0:
                    self.update_timestep()

                # Verifica se dt está nos limites (paranoia)
                if not (self.dt_min <= self.dt <= self.dt_max):
                    print(f"  AVISO: dt fora dos limites! Corrigindo {self.dt:.2e} → ", end="")
                    self.dt = self._validate_dt(self.dt, "main_loop_correction")
                    print(f"{self.dt:.2e}")

                # Avança no tempo
                self.t += self.dt
                self.step += 1

                # Registra convergência bem-sucedida
                self.consecutive_convergence += 1
                if self.consecutive_failures > 0:
                    self.consecutive_failures = 0
                    print("  ✓ Convergência restaurada!")

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

                elapsed = time.time() - start

                # Informação sobre o passo de tempo
                current_limit = self.get_current_saturation_limit()
                print(f"✓ Passo {self.step} concluído: t = {self.t:.4f}s | dt = {self.dt:.6e}s | Sat.Limit = {current_limit*100:.1f}% | tempo: {elapsed:.3f}s")

            else:
                # Falha: gerencia erro
                print("  ✗ Passo falhou - aplicando estratégia de recuperação")
                self.handle_convergence_failure()

                # Não avança no tempo, mas incrementa step para controle
                # Não incrementa self.step para não atrapalhar a contagem
                print(f"  ↻ Tentando novamente com dt = {self.dt:.6e}s")

        # Salva checkpoint final
        if self.checkpoint_interval > 0:
            self.save_checkpoint("checkpoint_final")

        # Finaliza
        self.close_xdmf_files()
        self.writer.write_summary(self.results)

        print("\n" + "="*90)
        print("SIMULAÇÃO CONCLUÍDA COM SUCESSO!")
        print("="*90)
        print(f"Total de passos: {self.step}")
        print(f"Tempo final: {self.t:.2f} s")
        print(f"Fase final: {self.phase} ({self.phase_params[self.phase]['name']})")
        print(f"Limite saturação final: {self.get_current_saturation_limit()*100:.1f}%")
        print(f"Método de dt usado: {self.dt_method}")
        print(f"dt final: {self.dt:.6e} s [validado: ✓]")
        print(f"Limites respeitados: [{self.dt_min:.2e}, {self.dt_max:.2e}] s")
        print(f"Falhas de convergência finais: {self.consecutive_failures}")
        print(f"Resultados salvos em: {self.output_dir}")
        print("="*90 + "\n")

# ==================== EXEMPLOS DE USO ====================

if __name__ == "__main__":

    # ========== EXEMPLO COM CONTROLE ADAPTATIVO DE SATURAÇÃO POR FASE ==========
    print("\n>>> TESTANDO CONTROLE ADAPTATIVO DE SATURAÇÃO POR FASE <<<\n")

    # Configurações de saturação para testar
    saturation_configs = [
        # Configuração padrão
        None,  # Usa padrão: {1: 0.05, 2: 0.1, 3: 0.15}
        
        # Configuração conservativa
        {1: 0.03, 2: 0.07, 3: 0.12},
        
        # Configuração agressiva
        {1: 0.08, 2: 0.15, 3: 0.25}
    ]

    config_names = ["padrao", "conservativo", "agressivo"]

    for i, sat_config in enumerate(saturation_configs):
        config_name = config_names[i]
        print(f"\n--- Testando configuração: {config_name} ---")

        solver_test = BrinkmanIMPESSolver(
            mesh_dir="/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/Arapua24/mesh",
            output_dir=f"/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/Arapua24/results_adaptive_sat_{config_name}",
            mu_w=1e-3,
            mu_o=1e-3,
            perm_darcy=100,
            dt=0.1,
            adaptive_dt=True,
            CFL_max=0.5,
            dt_min=1e-8,
            dt_max=2000.0,
            dt_method="robust_phases",
            checkpoint_interval=100,
            saturation_limits=sat_config  # NOVO PARÂMETRO
        )

        solver_test.run(T=1e5, impes_steps=256, save_interval=20)

        print(f"--- Configuração {config_name} finalizada ---\n")

    # ========== TESTE COM CONFIGURAÇÃO CUSTOMIZADA AVANÇADA ==========
    print("\n--- Teste Configuração Avançada ---")
    
    # Configuração muito específica
    custom_sat_limits = {
        1: 0.02,   # Fase 1: ultra conservativo (2%)
        2: 0.08,   # Fase 2: moderado (8%) 
        3: 0.20    # Fase 3: relaxado (20%)
    }

    solver_custom = BrinkmanIMPESSolver(
        mesh_dir="/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/Arapua24/mesh",
        output_dir="/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/Arapua24/results_custom_saturation",
        mu_w=1e-3,
        mu_o=1e-3,
        perm_darcy=100,
        dt=0.05,  # dt inicial menor
        adaptive_dt=True,
        CFL_max=0.3,  # CFL mais conservativo
        dt_min=1e-9,  # dt_min menor
        dt_max=1000.0,  # dt_max menor
        dt_method="robust_phases",
        checkpoint_interval=50,
        saturation_limits=custom_sat_limits
    )

    solver_custom.run(T=5e4, impes_steps=128, save_interval=10)

    print("\n*** TODOS OS TESTES DE CONTROLE ADAPTATIVO CONCLUÍDOS ***")
    print("Compare os resultados nos diretórios:")
    print("  - results_adaptive_sat_padrao/ (Limites padrão)")
    print("  - results_adaptive_sat_conservativo/ (Mais estável)")
    print("  - results_adaptive_sat_agressivo/ (Mais rápido)")
    print("  - results_custom_saturation/ (Configuração customizada)")
    print("\nAnálise esperada:")
    print("  - Conservativo: mais passos, mais estável, dt menores")
    print("  - Agressivo: menos passos, possíveis instabilidades, dt maiores")
    print("  - Padrão: equilíbrio entre estabilidade e performance")
    print("  - Custom: comportamento ultra-conservativo no início")
