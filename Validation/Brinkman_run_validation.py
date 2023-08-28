from Brinkman_Twophase_validation import *
from Brinkman_mono_validation import *

import os
import time

_folder_base = [
    "/home/tfk/Desktop/results/Validation/Brinkman_canal",
]


for i in _folder_base:
    try:
        os.mkdir(i)
    except OSError as error:
        print(error)

Nx = 50
Ny = 800
mu_w = 0.001
mu_o = 0.001
perm_matriz = 5e-9  # md
dt = 50
pin = (1 + 0.00825) / 98066.5
pout = 1 / 98066.5
# comentarios
# BrinkmanIMPES(Nx, _folder_base[0], mu_w, mu_o, perm_matriz, dt, pin, pout)
BrinkmanMonoBDM(perm_matriz, pin, pout, mu_w, Nx, Ny, _folder_base[0])
# BrinkmanIMPESGsmh(_folder_base[0], mu_w, mu_o, perm_matriz, dt)
