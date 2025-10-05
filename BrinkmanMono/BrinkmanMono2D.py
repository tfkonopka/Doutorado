"""
Brinkman Flow Solver Library
=============================
Biblioteca para simulação de escoamento em meios porosos com vugos
usando elementos finitos mistos (BDM) e equação de Brinkman.
"""

from fenics import *
import ufl_legacy as ufl
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional
import json


# Configuração de logging
def setup_logging(level='INFO', silent_fenics=True):
    """
    Configura o sistema de logging.
    
    Args:
        level: Nível de log ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        silent_fenics: Se True, suprime logs do FEniCS (FFC, UFL, DOLFIN)
    """
    # Configura logger principal
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(levelname)s - %(message)s'
    )
    
    # Suprime logs do FEniCS se solicitado
    if silent_fenics:
        # Silencia compilador de formas (FFC)
        logging.getLogger('FFC').setLevel(logging.WARNING)
        logging.getLogger('UFL').setLevel(logging.WARNING)
        logging.getLogger('UFL_LEGACY').setLevel(logging.WARNING)
        
        # Configura nível de log do FEniCS via set_log_level
        set_log_level(LogLevel.WARNING)  # ou LogLevel.ERROR para silenciar mais

# Inicializa com configuração padrão
setup_logging(level='INFO', silent_fenics=True)
logger = logging.getLogger(__name__)


# Constantes físicas
MILLIDARCY_TO_M2 = 9.86923e-16
KGF_CM2_TO_PA = 98066.5


@dataclass
class FluidProperties:
    """Propriedades do fluido."""
    viscosity: float  # Pa.s
    
    def __post_init__(self):
        if self.viscosity <= 0:
            raise ValueError("Viscosidade deve ser positiva")


@dataclass
class MediumProperties:
    """Propriedades do meio poroso."""
    permeability_md: float  # miliDarcy
    
    @property
    def permeability_m2(self) -> float:
        """Retorna permeabilidade em m²."""
        return self.permeability_md * MILLIDARCY_TO_M2
    
    def __post_init__(self):
        if self.permeability_md <= 0:
            raise ValueError("Permeabilidade deve ser positiva")


@dataclass
class BoundaryConditions:
    """Condições de contorno."""
    pressure_inlet: float  # Pa
    pressure_outlet: float  # Pa
    
    @property
    def pressure_drop(self) -> float:
        """Diferença de pressão."""
        return self.pressure_inlet - self.pressure_outlet
    
    def __post_init__(self):
        if self.pressure_inlet <= self.pressure_outlet:
            raise ValueError("Pressão de entrada deve ser maior que saída")


@dataclass
class NumericalParameters:
    """Parâmetros numéricos da simulação."""
    element_order: int = 1
    stabilization_parameter: float = 35.0
    
    def __post_init__(self):
        if self.element_order < 1:
            raise ValueError("Ordem dos elementos deve ser >= 1")
        if self.stabilization_parameter < 0:
            raise ValueError("Parâmetro de estabilização deve ser não-negativo")


class MeshLoader:
    """Carrega malha e marcadores de domínio/contorno."""
    
    def __init__(self, mesh_path: Path):
        """
        Inicializa o carregador de malha.
        
        Args:
            mesh_path: Caminho para o diretório contendo os arquivos de malha
        """
        self.mesh_path = Path(mesh_path)
        self._validate_mesh_files()
    
    def _validate_mesh_files(self):
        """Valida existência dos arquivos de malha."""
        required_files = ['mesh.xdmf', 'domains.xdmf', 'boundaries.xdmf']
        for file in required_files:
            if not (self.mesh_path / file).exists():
                raise FileNotFoundError(f"Arquivo {file} não encontrado em {self.mesh_path}")
    
    def load(self) -> Tuple[Mesh, MeshFunction, MeshFunction]:
        """
        Carrega malha e marcadores.
        
        Returns:
            Tupla contendo (mesh, domain_markers, boundary_markers)
        """
        logger.info(f"Carregando malha de {self.mesh_path}")
        
        mesh = Mesh()
        with XDMFFile(str(self.mesh_path / "mesh.xdmf")) as infile:
            infile.read(mesh)
        
        # Marcadores de domínio (2D)
        mvc_domains = MeshValueCollection("size_t", mesh, 2)
        with XDMFFile(str(self.mesh_path / "domains.xdmf")) as infile:
            infile.read(mvc_domains)
        domain_markers = cpp.mesh.MeshFunctionSizet(mesh, mvc_domains)
        
        # Marcadores de contorno (1D)
        mvc_boundaries = MeshValueCollection("size_t", mesh, 1)
        with XDMFFile(str(self.mesh_path / "boundaries.xdmf")) as infile:
            infile.read(mvc_boundaries)
        boundary_markers = cpp.mesh.MeshFunctionSizet(mesh, mvc_boundaries)
        
        logger.info(f"Malha carregada: {mesh.num_cells()} células")
        return mesh, domain_markers, boundary_markers


def tensor_jump(v, n):
    """
    Calcula o salto do tensor na interface.
    
    Args:
        v: Campo vetorial
        n: Vetor normal
        
    Returns:
        Salto do tensor
    """
    return ufl.outer(v, n)("+") + ufl.outer(v, n)("-")


class BrinkmanSolver:
    """Solver para equação de Brinkman com elementos BDM."""
    
    def __init__(
        self,
        mesh_path: Path,
        medium: MediumProperties,
        fluid: FluidProperties,
        bc: BoundaryConditions,
        numerical: Optional[NumericalParameters] = None
    ):
        """
        Inicializa o solver.
        
        Args:
            mesh_path: Caminho para os arquivos de malha
            medium: Propriedades do meio poroso
            fluid: Propriedades do fluido
            bc: Condições de contorno
            numerical: Parâmetros numéricos (usa valores padrão se None)
        """
        self.mesh_path = Path(mesh_path)
        self.medium = medium
        self.fluid = fluid
        self.bc = bc
        self.numerical = numerical or NumericalParameters()
        
        # Carrega malha
        mesh_loader = MeshLoader(self.mesh_path / "mesh")
        self.mesh, self.domain_markers, self.boundary_markers = mesh_loader.load()
        
        # Inicializa espaços de função
        self._setup_function_spaces()
        self._setup_measures()
        
        # Solução
        self.solution = None
    
    def _setup_function_spaces(self):
        """Configura espaços de elementos finitos."""
        logger.info("Configurando espaços de função")
        
        order = self.numerical.element_order
        V = FiniteElement("BDM", self.mesh.ufl_cell(), order)
        Q = FiniteElement("DG", self.mesh.ufl_cell(), order - 1)
        
        self.W = FunctionSpace(self.mesh, V * Q)
        logger.info(f"Graus de liberdade: {self.W.dim()}")
    
    def _setup_measures(self):
        """Configura medidas de integração."""
        self.ds = Measure("ds", domain=self.mesh, subdomain_data=self.boundary_markers)
        self.dx = Measure("dx", domain=self.mesh, subdomain_data=self.domain_markers)
    
    def _setup_boundary_conditions(self):
        """Configura condições de contorno Dirichlet."""
        zero_velocity = Constant((0.0, 0.0))
        
        # bc1: face Oeste (entrada) - comentada no original
        bc2 = DirichletBC(self.W.sub(0), zero_velocity, self.boundary_markers, 2)  # Norte
        # bc3: face Leste (saída) - comentada no original  
        bc4 = DirichletBC(self.W.sub(0), zero_velocity, self.boundary_markers, 4)  # Sul
        
        return [bc2, bc4]
    
    def _assemble_system(self):
        """Monta sistema linear."""
        logger.info("Montando sistema linear")
        
        # Funções de teste e trial
        (u, p) = TrialFunctions(self.W)
        (v, q) = TestFunctions(self.W)
        
        # Constantes
        mu = Constant(self.fluid.viscosity)
        k = Constant(self.medium.permeability_m2)
        pin = Constant(self.bc.pressure_inlet)
        pout = Constant(self.bc.pressure_outlet)
        f = Constant((0.0, 0.0))  # Força volumétrica
        
        # Geometria
        n = FacetNormal(self.mesh)
        h = CellDiameter(self.mesh)
        h_avg = ufl.Min(h("+"), h("-"))
        alpha = self.numerical.stabilization_parameter
        
        # Termo de estabilização (IP - Interior Penalty)
        stab = (
            mu * (alpha / h_avg) * inner(tensor_jump(u, n), tensor_jump(v, n)) * dS
            - mu * inner(avg(grad(u)), tensor_jump(v, n)) * dS
            - mu * inner(avg(grad(v)), tensor_jump(u, n)) * dS
        )
        
        # Forma bilinear
        a = (
            mu * inner(grad(u), grad(v)) * self.dx(1)  # Termo viscoso (vugos)
            + mu / k * inner(u, v) * self.dx(0)  # Termo de Darcy (matriz)
            - div(v) * p * self.dx(1)  # Acoplamento pressão-velocidade (vugos)
            - div(v) * p * self.dx(0)  # Acoplamento pressão-velocidade (matriz)
            + div(u) * q * self.dx(0)  # Incompressibilidade (matriz)
            + div(u) * q * self.dx(1)  # Incompressibilidade (vugos)
            + stab  # Estabilização
        )
        
        # Forma linear
        L = (
            inner(f, v) * self.dx(0)
            + inner(f, v) * self.dx(1)
            - pin * dot(v, n) * self.ds(1)  # Pressão entrada
            - pout * dot(v, n) * self.ds(3)  # Pressão saída
        )
        
        return a, L
    
    def solve(self):
        """
        Resolve o problema de Brinkman.
        
        Returns:
            Função solução (velocidade, pressão)
        """
        logger.info("Resolvendo sistema")
        
        a, L = self._assemble_system()
        bcs = self._setup_boundary_conditions()
        
        self.solution = Function(self.W)
        solve(a == L, self.solution, bcs)
        
        logger.info("Sistema resolvido com sucesso")
        return self.solution
    
    def save_solution(self, output_dir: Path):
        """
        Salva solução em arquivos PVD.
        
        Args:
            output_dir: Diretório de saída
        """
        if self.solution is None:
            raise RuntimeError("Solução não calculada. Execute solve() primeiro.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        u, p = self.solution.split()
        
        u_file = File(str(output_dir / "velocity.pvd"))
        u_file << u
        
        p_file = File(str(output_dir / "pressure.pvd"))
        p_file << p
        
        logger.info(f"Solução salva em {output_dir}")
    
    def compute_diagnostics(self) -> dict:
        """
        Calcula diagnósticos da solução.
        
        Returns:
            Dicionário com resultados (vazão, permeabilidade equivalente, etc.)
        """
        if self.solution is None:
            raise RuntimeError("Solução não calculada. Execute solve() primeiro.")
        
        logger.info("Calculando diagnósticos")
        
        u, p = self.solution.split()
        n = FacetNormal(self.mesh)
        
        # Áreas e comprimentos
        area_inlet = assemble(Constant(1.0) * self.ds(1))
        area_outlet = assemble(Constant(1.0) * self.ds(3))
        length = assemble(Constant(1.0) * self.ds(2))
        
        # Vazão
        flowrate = assemble(dot(u, n) * self.ds(1))
        
        # Pressões médias
        p_inlet_avg = assemble(p * self.ds(1)) / area_inlet
        p_outlet_avg = assemble(p * self.ds(3)) / area_outlet
        
        # Gradiente de pressão
        pressure_gradient = (p_inlet_avg - p_outlet_avg) / length
        
        # Permeabilidade equivalente (Lei de Darcy)
        k_equivalent_m2 = -(flowrate / (area_inlet * pressure_gradient)) * self.fluid.viscosity
        k_equivalent_md = k_equivalent_m2 / MILLIDARCY_TO_M2
        
        # Volumes
        vol_total = assemble(Constant(1.0) * self.dx)
        vol_matrix = assemble(Constant(1.0) * self.dx(0))
        vol_vugg = assemble(Constant(1.0) * self.dx(1))
        
        # Porosidade (assumindo 20% na matriz)
        vol_porous = vol_vugg + 0.2 * vol_matrix
        porosity_total = vol_porous / vol_total
        
        results = {
            'flowrate': float(flowrate),
            'pressure_gradient': float(pressure_gradient),
            'pressure_inlet_avg': float(p_inlet_avg),
            'pressure_outlet_avg': float(p_outlet_avg),
            'permeability_matrix_md': self.medium.permeability_md,
            'permeability_equivalent_md': float(k_equivalent_md),
            'permeability_ratio': float(k_equivalent_md / self.medium.permeability_md),
            'area_inlet': float(area_inlet),
            'length': float(length),
            'volume_total': float(vol_total),
            'volume_matrix': float(vol_matrix),
            'volume_vugg': float(vol_vugg),
            'volume_porous': float(vol_porous),
            'porosity_total': float(porosity_total)
        }
        
        return results
    
    def save_diagnostics(self, output_dir: Path, results: dict):
        """
        Salva diagnósticos em arquivo.
        
        Args:
            output_dir: Diretório de saída
            results: Dicionário com resultados
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Salva em formato texto com codificação UTF-8
        txt_path = output_dir / "diagnostics.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("BRINKMAN FLOW SIMULATION - DIAGNOSTICS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("PERMEABILITY\n")
            f.write("-" * 60 + "\n")
            f.write(f"Matrix permeability:      {results['permeability_matrix_md']:.4f} mD\n")
            f.write(f"Equivalent permeability:  {results['permeability_equivalent_md']:.4f} mD\n")
            f.write(f"Permeability ratio:       {results['permeability_ratio']:.4f}\n\n")
            
            f.write("FLOW\n")
            f.write("-" * 60 + "\n")
            f.write(f"Flowrate:                 {results['flowrate']:.6e} m³/s\n")
            f.write(f"Pressure gradient:        {results['pressure_gradient']:.6e} Pa/m\n")
            f.write(f"Inlet pressure (avg):     {results['pressure_inlet_avg']:.2f} Pa\n")
            f.write(f"Outlet pressure (avg):    {results['pressure_outlet_avg']:.2f} Pa\n\n")
            
            f.write("GEOMETRY\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total volume:             {results['volume_total']:.6e} m³\n")
            f.write(f"Matrix volume:            {results['volume_matrix']:.6e} m³\n")
            f.write(f"Vugg volume:              {results['volume_vugg']:.6e} m³\n")
            f.write(f"Porous volume:            {results['volume_porous']:.6e} m³\n")
            f.write(f"Total porosity:           {results['porosity_total']:.4f}\n")
        
        # Salva em formato JSON
        json_path = output_dir / "diagnostics.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Diagnósticos salvos em {output_dir}")


def run_simulation(
    mesh_path: Path,
    output_dir: Path,
    permeability_md: float,
    viscosity: float,
    pressure_inlet: float,
    pressure_outlet: float,
    stabilization_alpha: float = 35.0,
    verbose: bool = False
) -> dict:
    """
    Função de conveniência para executar simulação completa.
    
    Args:
        mesh_path: Caminho para diretório com malha
        output_dir: Diretório de saída
        permeability_md: Permeabilidade da matriz em miliDarcy
        viscosity: Viscosidade do fluido em Pa.s
        pressure_inlet: Pressão de entrada em Pa
        pressure_outlet: Pressão de saída em Pa
        stabilization_alpha: Parâmetro de estabilização
        verbose: Se True, mostra logs detalhados. Se False, apenas avisos e erros
        
    Returns:
        Dicionário com resultados diagnósticos
    """
    import time
    
    # Configura logging
    if verbose:
        setup_logging(level='INFO', silent_fenics=False)
    else:
        setup_logging(level='WARNING', silent_fenics=True)
    
    start = time.time()
    
    # Configura propriedades
    medium = MediumProperties(permeability_md=permeability_md)
    fluid = FluidProperties(viscosity=viscosity)
    bc = BoundaryConditions(pressure_inlet=pressure_inlet, pressure_outlet=pressure_outlet)
    numerical = NumericalParameters(stabilization_parameter=stabilization_alpha)
    
    # Cria solver e resolve
    solver = BrinkmanSolver(mesh_path, medium, fluid, bc, numerical)
    solver.solve()
    
    # Salva resultados
    solver.save_solution(output_dir)
    results = solver.compute_diagnostics()
    solver.save_diagnostics(output_dir, results)
    
    elapsed = time.time() - start
    logger.info(f"Simulação concluída em {elapsed:.2f} segundos")
    
    return results


if __name__ == "__main__":
    # Exemplo de uso - modo silencioso (padrão)
    mesh_path = Path("/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/mono_gamma_arapua17_V2")
    output_dir = Path("/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/mono_gamma_arapua17_V2/monophase")
    
    # Para execução silenciosa (apenas avisos e erros):
    results = run_simulation(
        mesh_path=mesh_path,
        output_dir=output_dir,
        permeability_md=100.0,
        viscosity=1.0,
        pressure_inlet=200000.0,
        pressure_outlet=100000.0,
        stabilization_alpha=35.0,
        verbose=False  # <-- Mude para True se quiser ver todos os logs
    )
    
    # Para execução com logs detalhados:
    # results = run_simulation(..., verbose=True)
    
    print("\n" + "="*60)
    print("RESULTADOS DA SIMULAÇÃO")
    print("="*60)
    print(f"Vazão: {results['flowrate']:.6e} m³/s")
    print(f"Permeabilidade equivalente: {results['permeability_equivalent_md']:.2f} mD")
    print(f"Razão de permeabilidade: {results['permeability_ratio']:.2f}")
    print(f"Porosidade total: {results['porosity_total']:.4f}")
    print("="*60)