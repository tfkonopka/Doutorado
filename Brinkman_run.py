# from brinkman_biphase_IMPES_gsmh import *  # funcvional
# from Brinkman_Twophase_sintetico_deltaP import *
from brinkman_biphase_IMPES_gsmh import *

import os
import time

_folder_base = [
    "/home/tfkonopka/results/Arapua17_mu_20cp",
]


for i in _folder_base:
    try:
        os.mkdir(i)
    except OSError as error:
        print(error)

Nx = 20
mu_w = 0.001
mu_o = 0.02
perm_matriz = 100  # md
dt = 20
pin = 2
pout = 1
# comentarios
# BrinkmanIMPES(Nx, _folder_base[0], mu_w, mu_o, perm_matriz, dt, pin, pout)
BrinkmanIMPESGsmh(_folder_base[0], mu_w, mu_o, perm_matriz, dt)
