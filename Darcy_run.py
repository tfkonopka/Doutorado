# from fenics import *
import matplotlib.pyplot as plt
import os
from Darcy_Twophase_Sintetico_deltaP import *


_folder_base = [
    "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/vugg_sintetico/sens_mu_variavel/Darcy_canal_muo1",
]


for i in _folder_base:
    try:
        os.mkdir(i)
    except OSError as error:
        print(error)


Nx = 50
mu_w = 0.001
mu_o = 0.001
perm_matriz = 100  # md
perm_vugg = [8000]
dt = 200

pin = 2
pout = 1
i = 0

start_time = time.time()


# for i in range(len(perm_vugg)):
#     DarcyIMPES(Nx, _folder_base[i], mu_w, mu_o, perm_matriz,perm_vugg[i], dt)

# DarcyIMPES(Nx, _folder_base[i], mu_w, mu_o, perm_matriz, perm_vugg[i], dt)

DarcyIMPESRT(Nx, _folder_base[i], mu_w, mu_o, perm_matriz, perm_vugg[i], dt)

print("Total time simulation ---%s  seconds--" % (time.time() - start_time))
