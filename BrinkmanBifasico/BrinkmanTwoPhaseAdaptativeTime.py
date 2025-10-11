"""
Biblioteca para simulação de fluxo bifásico usando método Brinkman-IMPES
Versão Final - Completa e Testada
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
                 checkpoint_interval: int = 0):
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
        self.checkpoint_interval = checkpoint_interval
        self.t = 0.0
        self.step = 0
        
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
        
        print(f"  dt atual={self.dt:.6e} s")
        print(f"  dt atual={self.dt:.6e} s")
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
    
    def compute_adaptive_timestep(self):
        """
        Calcula passo de tempo adaptativo baseado no número de CFL
        CFL = u * dt / h
        """
        if not self.adaptive_dt:
            return self.dt
        
        (u_, p_) = self.U.split()
        
        # Velocidade máxima no domínio
        u_array = u_.vector().get_local()
        u_magnitude = np.sqrt(u_array[::2]**2 + u_array[1::2]**2)
        u_max = np.max(u_magnitude)
        
        if u_max < 1e-15:
            # Velocidade muito pequena, usa dt máximo
            dt_new = self.dt_max
        else:
            # Tamanho mínimo da célula
            h_min = self.mesh.hmin()
            
            # Calcula dt baseado em CFL
            dt_new = self.CFL_max * h_min / u_max
            
            # Limita variação do dt (máx 50% de mudança por passo)
            dt_new = max(min(dt_new, 1.5 * self.dt), 0.75 * self.dt)
            
            # Aplica limites globais
            dt_new = max(min(dt_new, self.dt_max), self.dt_min)
        
        return dt_new
    
    def update_timestep(self):
        """Atualiza o passo de tempo se adaptativo estiver ativado"""
        if self.adaptive_dt:
            dt_old = self.dt
            self.dt = self.compute_adaptive_timestep()
            
            if abs(self.dt - dt_old) > 1e-10:
                print(f"  -> dt ajustado: {dt_old:.6e} → {self.dt:.6e} s")
                # Precisa remontar o sistema de saturação com novo dt
                self.assemble_saturation_system()
    
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
            
            # Carrega histórico de resultados
            with open(results_file, 'r') as f:
                self.results = json.load(f)
            
            print(f"  Checkpoint carregado com sucesso!")
            print(f"  Reiniciando do passo {self.step}, t = {self.t:.4f} s")
            
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
        print(f"Resultados salvos em: {self.output_dir}")
        print("="*60 + "\n")


# ==================== EXEMPLO DE USO ====================
if __name__ == "__main__":
    
    # ========== EXEMPLO 1: SEM checkpoints (padrão) ==========
    print("\n>>> EXEMPLO 1: Simulação SEM checkpoints <<<\n")
    
    solver1 = BrinkmanIMPESSolver(
        mesh_dir="./mesh",
        output_dir="./results",
        mu_w=1e-3,
        mu_o=10e-3,
        perm_darcy=100,
        dt=0.1,
        inlet_velocity=1.0e-6,
        checkpoint_interval=0  # 0 = sem checkpoints (padrão)
    )
    
    # solver1.run(T=100.0, impes_steps=5, save_interval=10)
    
    
    # ========== EXEMPLO 2: COM checkpoints automáticos ==========
    print("\n>>> EXEMPLO 2: Simulação COM checkpoints <<<\n")
    
    solver2 = BrinkmanIMPESSolver(
        mesh_dir="./mesh",
        output_dir="./results_checkpoint",
        mu_w=1e-3,
        mu_o=10e-3,
        perm_darcy=100,
        dt=0.1,
        inlet_velocity=1.0e-6,
        checkpoint_interval=100  # ← Salva checkpoint a cada 100 passos
    )
    
    # solver2.run(T=100.0, impes_steps=5, save_interval=10)
    
    
    # ========== EXEMPLO 3: REINICIAR de checkpoint ==========
    print("\n>>> EXEMPLO 3: REINICIAR simulação de checkpoint <<<\n")
    
    solver3 = BrinkmanIMPESSolver(
        mesh_dir="./mesh",
        output_dir="./results_checkpoint",
        mu_w=1e-3,
        mu_o=10e-3,
        perm_darcy=100,
        dt=0.1,
        inlet_velocity=1.0e-6,
        checkpoint_interval=100
    )
    
    # Reinicia do último checkpoint
    # solver3.run(
    #     T=200.0,  # Continua até t=200
    #     impes_steps=5,
    #     save_interval=10,
    #     restart_from_checkpoint="checkpoint_final"  # ← Reinicia daqui
    # )
    
    
    # ========== EXEMPLO 4: Configuração COMPLETA ==========
    print("\n>>> EXEMPLO 4: Configuração COMPLETA com tudo ativado <<<\n")
    
    solver_completo = BrinkmanIMPESSolver(
        mesh_dir="/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/Arapua24/mesh",
        output_dir="/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/Arapua24/results",
        mu_w=1e-3,
        mu_o=1e-3,
        perm_darcy=100,
        dt=0.1,
        phi=0.2,
        inlet_velocity=5.0e-6,
        pin=200000,
        pout=100000,
        adaptive_dt=True,           # ← dt adaptativo
        CFL_max=0.5,
        dt_min=50,
        dt_max=500,
        checkpoint_interval=500     # ← Checkpoint a cada 500 passos
    )
    
    # Executar simulação nova
    solver_completo.run(
        T=100.0,
        impes_steps=256,
        save_interval=10
    )
    
    # OU reiniciar de checkpoint se a simulação foi interrompida:
    # solver_completo.run(
    #     T=200.0,
    #     impes_steps=256,
    #     save_interval=10,
    #     restart_from_checkpoint="checkpoint_step_500"  # ou "checkpoint_final"
    # )