from FIAT import finite_element, dual_set, reference_element

# Define o triângulo de referência
triangle = reference_element.UFCTriangle

# Define o elemento BDM1 para um triângulo
element = finite_element.FiniteElement("BDM", triangle, 1)

# Obtém o conjunto de funções de forma
basis_functions = element.dual_basis()

# Imprime as funções de forma
for i, phi in enumerate(basis_functions):
    print(f"Função de Forma {i+1}: {phi}")
