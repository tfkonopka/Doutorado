from Brinkman_Twophase_validation import *

import os
import time

_folder_base = [
    "/home/tfk/Desktop/results/Validation/Brinkman_M1",
]


for i in _folder_base:
    try:
        os.mkdir(i)
    except OSError as error:
        print(error)

Nx = 500
mu_w = 0.001
mu_o = 0.001
perm_matriz = 100  # md
dt = 50
pin = 2
pout = 1
# comentarios
BrinkmanIMPES(Nx, _folder_base[0], mu_w, mu_o, perm_matriz, dt, pin, pout)
# BrinkmanIMPESGsmh(_folder_base[0], mu_w, mu_o, perm_matriz, dt)
