# from Brinkman_Twophase_sintetico_deltaP import *
# from Brinkman_Twophase_sintetico_vugg_3 import *

from brinkman_biphase_IMPES_gsmh import *

# from Brinkman_mono_sintetico import *
# from brinkman_mono_arapua import *

import os
import time

_folder_base = [
    "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/RD/DR_am_8_mesh4",
]


for i in _folder_base:
    try:
        os.mkdir(i)
    except OSError as error:
        print(error)

Nx = 120
Ny = Nx
mu_w = 0.001
mu_o = 0.001
perm_matriz = 100  # md
dt = 2
pin = 2
pout = 1
# comentarios


start_time = time.time()
BrinkmanIMPESGsmh(_folder_base[0], mu_w, mu_o, perm_matriz, dt)
# BrinkmanIMPES(Nx, _folder_base[0], mu_w, mu_o, perm_matriz, dt, pin, pout)
print(time.time() - start_time)

# BrinkmanMonoBDM(perm_matriz, pin, pout, mu_w, Nx, Ny, _folder_base[0])
# BrinkmanIMPESGsmh(_folder_base[0], mu_w, mu_o, perm_matriz, dt)
# BrinkmanIMPES(Nx, _folder_base[0], mu_w, mu_o, perm_matriz, dt, pin, pout)
