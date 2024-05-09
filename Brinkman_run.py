# from Brinkman_Twophase_sintetico_deltaP import *

# from Brinkman_Twophase_sintetico_vugg_3 import *

from Brinkman_Twophase_sintetico_validacao2 import *

# from Brinkman_mono_sintetico import *
# from brinkman_mono_arapua import *

import os
import time

_folder_base = [
    "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/validacaoCocliteHperm2_Mu1e10",
    # "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/vugg_sintetico/five_spot",
]


for i in _folder_base:
    try:
        os.mkdir(i)
    except OSError as error:
        print(error)

Nx = 10
Ny = 10
mu_w = 1
mu_o = 1
perm_matriz = 1  # md
dt = 1e-8
pin = 2
pout = 1
# comentarios


start_time = time.time()
# BrinkmanIMPES(_folder_base[0], mu_w, mu_o, perm_matriz, dt)
BrinkmanIMPES(Nx, _folder_base[0], mu_w, mu_o, perm_matriz, dt, pin, pout)
print(time.time() - start_time)

# BrinkmanMonoBDM(perm_matriz, pin, pout, mu_w, Nx, Ny, _folder_base[0])
# BrinkmanIMPESGsmh(_folder_base[0], mu_w, mu_o, perm_matriz, dt)
# BrinkmanIMPES(Nx, _folder_base[0], mu_w, mu_o, perm_matriz, dt, pin, pout)
