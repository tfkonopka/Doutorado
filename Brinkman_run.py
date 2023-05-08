
from Brinkman_Twophase_sintetico_vugg_3 import *

import os
import time

_folder_base = [
    "/home/tfkonopka/results/Brinkman_Sintetico_Vugg_3",
]


for i in _folder_base:
    try:
        os.mkdir(i)
    except OSError as error:
        print(error)

Nx = 100
mu_w = 0.001
mu_o = 0.001
perm_matriz = 100  # md
dt = 200
pin = 2
pout = 1
# comentarios
BrinkmanIMPES(Nx, _folder_base[0], mu_w, mu_o, perm_matriz, dt, pin, pout)
# BrinkmanIMPESGsmh(_folder_base[0], mu_w, mu_o, perm_matriz, dt)
