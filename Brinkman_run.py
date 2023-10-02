# from Brinkman_Twophase_sintetico_deltaP import *
from Brinkman_mono_sintetico import *

import os
import time

_folder_base = [
    "/home/tfk/Desktop/results/Validation/Mono_Mesh_Size/vug16_Mesh20",
]


for i in _folder_base:
    try:
        os.mkdir(i)
    except OSError as error:
        print(error)

Nx = 20
Ny = Nx
mu_w = 0.001
mu_o = 0.001
perm_matriz = 100  # md
dt = 50
pin = 2
pout = 1
# comentarios
# BrinkmanIMPES(Nx, _folder_base[0], mu_w, mu_o, perm_matriz, dt, pin, pout)

BrinkmanMonoBDM(perm_matriz, pin, pout, mu_w, Nx, Ny, _folder_base[0])
# BrinkmanIMPESGsmh(_folder_base[0], mu_w, mu_o, perm_matriz, dt)
