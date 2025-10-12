import numpy as np
from fenics import *
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import time as time_module

class BrinkmanTwoPhaseAdaptiveTime:
    def __init__(self, mesh, fluid_properties, boundary_conditions, 
                 initial_conditions, numerical_params):
        
        # Malha e espaços funcionais
        self.mesh = mesh
        self.dim = mesh.topology().dim()
        
        # Espaço misto para velocidade e pressão (P2-P1 ou P1-P0)
        V_elem = VectorElement("CG", mesh.ufl_cell(), 2)
        P_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
        self.W = FunctionSpace(mesh, V_elem * P_elem)
        
        # Espaço DG para saturação
        self.V_s = FunctionSpace(mesh, "DG", 1)
        
        # Funções
        self.U = Function(self.W)  # Velocidade e pressão atual
        self.U_old = Function(self.W)  # Velocidade e pressão anterior
        self.s = Function(self.V_s)  # Saturação atual
        self.s_old = Function(self.V_s)  # Saturação anterior
        
        # Funções teste
        (self.v, self.q) = TestFunctions(self.W)
        self.phi = TestFunction(self.V_s)
        
        # Propriedades dos fluidos
        self.rho_w = Constant(fluid_properties.get('rho_w', 1000.0))  # densidade água
        self.rho_o = Constant(fluid_properties.get('rho_o', 800.0))   # densidade óleo
        self.mu_w = Constant(fluid_properties.get('mu_w', 1e-3))      # viscosidade água
        self.mu_o = Constant(fluid_properties.get('mu_o', 5e-3))      # viscosidade óleo
        self.porosity = Constant(fluid_properties.get('phi', 0.3))    # porosidade
        self.permeability = Constant(fluid_properties.get('k', 1e-12)) # permeabilidade
        
        # Condições de contorno
        self.bcs_flow = boundary_conditions.get('flow', [])
        self.bcs_saturation = boundary_conditions.get('saturation', [])
        
        # Condições iniciais
        self.s.interpolate(initial_conditions.get('saturation', Constant(0.0)))
        self.s_old.assign(self.s)
        
        # Parâmetros numéricos
        self.dt = numerical_params.get('dt', 1e-4)
        self.dt_min = numerical_params.get('dt_min', 1e-8)
        self.dt_max = numerical_params.get('dt_max', 1e-2)
        self.adaptive_dt = numerical_params.get('adaptive_dt', True)
        
        # Parâmetros da nova estratégia adaptativa
        self.phase = 1  # Fase atual (1: início, 2: desenvolvimento, 3: regime)
        self.step = 0
        self.consecutive_convergence = 0
        self.consecutive_failures = 0
        self.last_dt_values = []  # Histórico dos últimos dt
        
        # Parâmetros por fase
        self.phase_params = {
            1: {'cfl_max': 0.5, 'max_steps': 20, 'dt_variation': 1.1},
            2: {'cfl_max': 0.8, 'max_steps': 100, 'dt_variation': 1.2},
            3: {'cfl_max': 1.0, 'max_steps': float('inf'), 'dt_variation': 1.2}
        }
        
        # Thresholds
        self.max_saturation_change = 0.1
        self.convergence_threshold = 15  # máximo de iterações não-lineares
        
        # Armazenamento de resultados
        self.results = {
            'time': [],
            'dt': [],
            'saturation_max': [],
            'saturation_mean': [],
            'velocity_max': [],
            'phase': [],
            'convergence_info': []
        }
        
        # Solver não-linear
        self.newton_solver = NewtonSolver()
        self.newton_solver.parameters['absolute_tolerance'] = 1e-8
        self.newton_solver.parameters['relative_tolerance'] = 1e-7
        self.newton_solver.parameters['maximum_iterations'] = 25
        self.newton_solver.parameters['relaxation_parameter'] = 1.0
        
        print("BrinkmanTwoPhaseAdaptiveTime inicializado com estratégia robusta")
        print(f"  dt inicial: {self.dt:.2e}")
        print(f"  Fase inicial: {self.phase} (CFL_max = {self.phase_params[self.phase]['cfl_max']})")

    def relative_permeability(self, s):
        """Permeabilidades relativas usando modelo de Corey"""
        # Parâmetros típicos
        s_wr = 0.2  # saturação residual de água
        s_or = 0.2  # saturação residual de óleo
        
        # Saturação normalizada
        s_n = (s - s_wr) / (1 - s_wr - s_or)
        s_n = max_value(min_value(s_n, 1.0), 0.0)
        
        # Permeabilidades relativas (Corey)
        kr_w = s_n**2
        kr_o = (1 - s_n)**2
        
        return kr_w, kr_o

    def effective_properties(self, s):
        """Propriedades efetivas baseadas na saturação"""
        kr_w, kr_o = self.relative_permeability(s)
        
        # Mobilidades
        lambda_w = kr_w / self.mu_w
        lambda_o = kr_o / self.mu_o
        lambda_t = lambda_w + lambda_o
        
        # Densidade efetiva
        rho_eff = s * self.rho_w + (1 - s) * self.rho_o
        
        return lambda_t, rho_eff, lambda_w, lambda_o

    def momentum_equation(self):
        """Equação de momentum com lei de Darcy-Brinkman"""
        (u, p) = split(self.U)
        (u_old, p_old) = split(self.U_old)
        
        # Propriedades efetivas na saturação atual
        lambda_t, rho_eff, lambda_w, lambda_o = self.effective_properties(self.s)
        
        # Permeabilidade efetiva
        K_eff = self.permeability * lambda_t
        
        # Equação de Darcy-Brinkman
        F_momentum = (
            # Termo de permeabilidade
            inner(u, self.v) / K_eff * dx
            
            # Gradiente de pressão
            + inner(grad(p), self.v) * dx
            
            # Termo gravitacional
            - rho_eff * inner(Constant((0, -9.81)), self.v) * dx
            
            # Continuidade
            + div(u) * self.q * dx
        )
        
        return F_momentum

    def transport_equation(self):
        """Equação de transporte para saturação usando DG"""
        (u, p) = split(self.U)
        
        # Fluxo fracionário
        kr_w, kr_o = self.relative_permeability(self.s)
        lambda_w = kr_w / self.mu_w
        lambda_o = kr_o / self.mu_o
        lambda_t = lambda_w + lambda_o
        
        f_w = lambda_w / (lambda_t + 1e-12)  # Evita divisão por zero
        
        # Velocidade total (Darcy)
        velocity = u
        
        # Forma variacional DG para transporte
        n = FacetNormal(self.mesh)
        h = CellDiameter(self.mesh)
        
        # Termo temporal
        F_transport = (
            (self.s - self.s_old) / self.dt * self.phi * dx
        )
        
        # Termo convectivo (upwind)
        F_transport += (
            - f_w * inner(velocity, grad(self.phi)) * dx
            + f_w('+') * max_value(inner(velocity('+'), n('+')), 0) * jump(self.phi) * dS
            + f_w('-') * max_value(inner(velocity('-'), n('-')), 0) * jump(self.phi) * dS
        )
        
        # Estabilização DG
        alpha = 0.1  # parâmetro de estabilização
        F_transport += alpha * avg(h) * jump(grad(f_w)) * jump(grad(self.phi)) * dS
        
        return F_transport

    def solve_flow_step(self):
        """Resolve a equação de flow (velocidade e pressão)"""
        F = self.momentum_equation()
        
        # Resolve sistema não-linear
        solve(F == 0, self.U, self.bcs_flow,
              solver_parameters={'newton_solver': {
                  'absolute_tolerance': 1e-8,
                  'relative_tolerance': 1e-7,
                  'maximum_iterations': 15
              }})
        
        return True

    def solve_transport_step(self):
        """Resolve a equação de transporte (saturação)"""
        F_transport = self.transport_equation()
        
        try:
            # Solver linear para equação de transporte
            solve(F_transport == 0, self.s, self.bcs_saturation)
            
            # Limita saturação entre 0 e 1
            s_array = self.s.vector().get_local()
            s_array = np.clip(s_array, 0.0, 1.0)
            self.s.vector().set_local(s_array)
            self.s.vector().apply('insert')
            
            return True
            
        except Exception as e:
            print(f"    Erro na equação de transporte: {e}")
            return False

    def compute_adaptive_timestep_local(self):
        """Método CFL local - critério principal da nova estratégia"""
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
            return self.dt_max
        
        # Usa percentil 95 para robustez
        cfl_max = np.percentile(cfl_array, 95)
        
        if cfl_max < 1e-12:
            return self.dt_max
        
        # CFL_max baseado na fase atual
        cfl_target = self.phase_params[self.phase]['cfl_max']
        dt_new = cfl_target / cfl_max
        
        return dt_new

    def compute_saturation_check(self):
        """Verifica se a variação de saturação está controlada"""
        if len(self.results['time']) < 1:
            return self.dt_max
        
        # Variação máxima de saturação
        s_array = self.s.vector().get_local()
        s_old_array = self.s_old.vector().get_local()
        
        max_change = np.max(np.abs(s_array - s_old_array))
        
        # Se variação é muito grande, limita dt
        if max_change > self.max_saturation_change:
            # Reduz dt proporcionalmente
            factor = self.max_saturation_change / max_change
            return self.dt * factor * 0.8  # margem de segurança
        
        return self.dt_max  # Não limita

    def compute_adaptive_timestep(self):
        """Nova estratégia robusta de passo de tempo adaptativo"""
        if not self.adaptive_dt:
            return self.dt
        
        # Critério principal: CFL local
        dt_cfl = self.compute_adaptive_timestep_local()
        
        # Critério de segurança: variação de saturação
        dt_sat_check = self.compute_saturation_check()
        
        # Usa o mais restritivo
        dt_new = min(dt_cfl, dt_sat_check)
        
        # Aplicar limitação de variação baseada na fase
        max_variation = self.phase_params[self.phase]['dt_variation']
        dt_new = max(min(dt_new, max_variation * self.dt), self.dt / max_variation)
        
        # Limites globais
        dt_new = max(min(dt_new, self.dt_max), self.dt_min)
        
        return dt_new

    def check_phase_transition(self):
        """Verifica se deve mudar de fase"""
        if self.phase == 1:
            # Fase 1 → 2: após passos bem-sucedidos
            if (self.step >= self.phase_params[1]['max_steps'] and 
                self.consecutive_convergence >= 10):
                self.phase = 2
                print(f"  → Transição para Fase 2 (desenvolvimento) - CFL_max = {self.phase_params[2]['cfl_max']}")
                
        elif self.phase == 2:
            # Fase 2 → 3: quando saturação está estabelecida
            if (self.step >= self.phase_params[2]['max_steps'] or
                (len(self.results['saturation_mean']) >= 10 and
                 np.std(self.results['saturation_mean'][-10:]) < 0.01)):
                self.phase = 3
                print(f"  → Transição para Fase 3 (regime) - CFL_max = {self.phase_params[3]['cfl_max']}")

    def handle_convergence_failure(self):
        """Gerencia falhas de convergência"""
        self.consecutive_failures += 1
        self.consecutive_convergence = 0
        
        if self.consecutive_failures == 1:
            # Primeira falha: reduz dt moderadamente
            self.dt *= 0.7
            print(f"    Falha de convergência. Reduzindo dt para {self.dt:.2e}")
            
        elif self.consecutive_failures <= 3:
            # Falhas múltiplas: redução mais agressiva
            self.dt *= 0.5
            print(f"    Falha {self.consecutive_failures}. dt = {self.dt:.2e}")
            
        else:
            # Muitas falhas: volta para dt_min e Fase 1
            self.dt = self.dt_min
            self.phase = 1
            self.consecutive_failures = 0
            print(f"    Reset para Fase 1. dt = {self.dt:.2e}")
        
        # Garante que não fica abaixo do mínimo
        self.dt = max(self.dt, self.dt_min)

    def update_timestep(self):
        """Atualiza passo de tempo com nova estratégia"""
        if not self.adaptive_dt:
            return
        
        # Calcula novo dt
        dt_new = self.compute_adaptive_timestep()
        
        # Atualiza histórico
        self.last_dt_values.append(dt_new)
        if len(self.last_dt_values) > 5:
            self.last_dt_values.pop(0)
        
        # Atualiza dt
        self.dt = dt_new
        
        # Verifica transição de fase
        self.check_phase_transition()
        
        print(f"  Fase {self.phase}, dt = {self.dt:.2e} (CFL_max = {self.phase_params[self.phase]['cfl_max']})")

    def save_results(self, t):
        """Salva resultados da simulação"""
        # Calcula estatísticas
        s_array = self.s.vector().get_local()
        (u_, p_) = self.U.split()
        
        # Projeta velocidade para calcular magnitude
        DG0 = FunctionSpace(self.mesh, "DG", 0)
        u_mag = project(sqrt(inner(u_, u_)), DG0)
        u_array = u_mag.vector().get_local()
        
        # Armazena resultados
        self.results['time'].append(t)
        self.results['dt'].append(self.dt)
        self.results['saturation_max'].append(np.max(s_array))
        self.results['saturation_mean'].append(np.mean(s_array))
        self.results['velocity_max'].append(np.max(u_array))
        self.results['phase'].append(self.phase)
        self.results['convergence_info'].append({
            'consecutive_convergence': self.consecutive_convergence,
            'consecutive_failures': self.consecutive_failures
        })

    def run(self, T, impes_steps=10, save_interval=10):
        """Executa simulação IMPES com nova estratégia adaptativa"""
        print(f"\nIniciando simulação IMPES com estratégia robusta")
        print(f"Tempo final: {T}, IMPES steps: {impes_steps}")
        
        t = 0.0
        
        # Salva condição inicial
        self.save_results(t)
        
        # Resolve campo inicial de velocidade
        print(f"Resolvendo campo inicial de velocidade...")
        self.solve_flow_step()
        
        start_time = time_module.time()
        
        while t < T:
            self.step += 1
            
            print(f"\n=== Passo {self.step}, t = {t:.4e} ===")
            
            # Atualiza passo de tempo
            self.update_timestep()
            
            # Verificar se dt não ficou muito pequeno
            if self.dt < self.dt_min * 1.1:
                print(f"  Aviso: dt próximo do mínimo ({self.dt:.2e})")
            
            step_success = True
            
            # IMPES: resolve velocidade uma vez a cada impes_steps
            if self.step % impes_steps == 1:
                print("  Resolvendo campo de velocidade...")
                try:
                    flow_success = self.solve_flow_step()
                    if flow_success:
                        self.consecutive_convergence += 1
                        self.consecutive_failures = 0
                    else:
                        step_success = False
                except Exception as e:
                    print(f"    Erro no campo de velocidade: {e}")
                    step_success = False
            
            # Resolve transporte
            if step_success:
                print("  Resolvendo transporte...")
                try:
                    transport_success = self.solve_transport_step()
                    if not transport_success:
                        step_success = False
                except Exception as e:
                    print(f"    Erro no transporte: {e}")
                    step_success = False
            
            # Verifica sucesso do passo
            if step_success:
                # Sucesso: avança no tempo
                t += self.dt
                
                # Atualiza soluções
                self.U_old.assign(self.U)
                self.s_old.assign(self.s)
                
                # Salva resultados
                if self.step % save_interval == 0:
                    self.save_results(t)
                    print(f"  Resultados salvos (s_max = {self.results['saturation_max'][-1]:.4f})")
                
                # Reset contador de falhas
                if self.consecutive_failures > 0:
                    self.consecutive_failures = 0
                    print("  Convergência restaurada!")
                
            else:
                # Falha: gerencia erro
                print("  ✗ Passo falhou")
                self.handle_convergence_failure()
                
                # Não avança no tempo, tenta novamente
                continue
            
            # Informações de progresso
            if self.step % 50 == 0:
                elapsed = time_module.time() - start_time
                progress = t / T * 100
                print(f"\n  Progresso: {progress:.1f}% ({t:.2e}/{T:.2e})")
                print(f"  Tempo decorrido: {elapsed:.1f}s")
                print(f"  Saturação média: {np.mean(self.s.vector().get_local()):.4f}")
        
        # Salva resultado final
        self.save_results(t)
        
        total_time = time_module.time() - start_time
        print(f"\n=== Simulação Concluída ===")
        print(f"Tempo total: {total_time:.1f}s")
        print(f"Passos realizados: {self.step}")
        print(f"Fase final: {self.phase}")
        print(f"dt final: {self.dt:.2e}")

    def plot_results(self):
        """Plotar resultados da simulação"""
        if len(self.results['time']) < 2:
            print("Dados insuficientes para plotar")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Evolução do passo de tempo
        axes[0,0].semilogy(self.results['time'], self.results['dt'])
        axes[0,0].set_xlabel('Tempo')
        axes[0,0].set_ylabel('Passo de tempo (dt)')
        axes[0,0].set_title('Evolução do Passo de Tempo')
        axes[0,0].grid(True)
        
        # Fases da simulação
        phases = np.array(self.results['phase'])
        for phase in [1, 2, 3]:
            mask = phases == phase
            if np.any(mask):
                times = np.array(self.results['time'])[mask]
                dts = np.array(self.results['dt'])[mask]
                axes[0,0].semilogy(times, dts, 'o', markersize=3, 
                                 label=f'Fase {phase}')
        axes[0,0].legend()
        
        # Saturação máxima
        axes[0,1].plot(self.results['time'], self.results['saturation_max'])
        axes[0,1].set_xlabel('Tempo')
        axes[0,1].set_ylabel('Saturação Máxima')
        axes[0,1].set_title('Evolução da Saturação Máxima')
        axes[0,1].grid(True)
        
        # Saturação média
        axes[1,0].plot(self.results['time'], self.results['saturation_mean'])
        axes[1,0].set_xlabel('Tempo')
        axes[1,0].set_ylabel('Saturação Média')
        axes[1,0].set_title('Evolução da Saturação Média')
        axes[1,0].grid(True)
        
        # Velocidade máxima
        axes[1,1].semilogy(self.results['time'], self.results['velocity_max'])
        axes[1,1].set_xlabel('Tempo')
        axes[1,1].set_ylabel('Velocidade Máxima')
        axes[1,1].set_title('Evolução da Velocidade Máxima')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.show()

# ==================== EXEMPLO DE USO ====================

if __name__ == "__main__":
    # Criar malha
    from mshr import Rectangle, generate_mesh
    domain = Rectangle(Point(0, 0), Point(1, 1))
    mesh = generate_mesh(domain, 50)
    
    # Propriedades dos fluidos
    fluid_props = {
        'rho_w': 1000.0,    # kg/m³
        'rho_o': 800.0,     # kg/m³ 
        'mu_w': 1e-3,       # Pa·s
        'mu_o': 5e-3,       # Pa·s
        'phi': 0.3,         # porosidade
        'k': 1e-12          # permeabilidade (m²)
    }
    
    # Condições de contorno para velocidade/pressão
    def inlet_velocity(x, on_boundary):
        return on_boundary and near(x[0], 0.0)
    
    def outlet_pressure(x, on_boundary):
        return on_boundary and near(x[0], 1.0)
    
    V_elem = VectorElement("CG", mesh.ufl_cell(), 2)
    P_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, V_elem * P_elem)
    
    # Velocidade na entrada
    inlet_vel = Expression(("0.01", "0.0"), degree=2)
    bc_inlet = DirichletBC(W.sub(0), inlet_vel, inlet_velocity)
    
    # Pressão na saída
    bc_outlet = DirichletBC(W.sub(1), Constant(0.0), outlet_pressure)
    
    boundary_conditions = {
        'flow': [bc_inlet, bc_outlet],
        'saturation': []
    }
    
    # Condições iniciais
    class InitialSaturation(UserExpression):
        def eval(self, values, x):
            if x[0] < 0.1:  # Região de injeção
                values[0] = 1.0
            else:
                values[0] = 0.0
        
        def value_shape(self):
            return ()
    
    initial_conditions = {
        'saturation': InitialSaturation(degree=1)
    }
    
    # Parâmetros numéricos com nova estratégia
    numerical_params = {
        'dt': 1e-4,        # passo inicial
        'dt_min': 1e-8,    # passo mínimo
        'dt_max': 1e-2,    # passo máximo
        'adaptive_dt': True
    }
    
    # Criar solver
    solver = BrinkmanTwoPhaseAdaptiveTime(
        mesh, fluid_props, boundary_conditions, 
        initial_conditions, numerical_params
    )
    
    # Executar simulação
    solver.run(T=1e5, impes_steps=256, save_interval=20)
    
    # Plotar resultados
    solver.plot_results()
