# from Brinkman_Twophase_sintetico_deltaP import *

# from brinkman_biphase_IMPES_gsmh import *

# from Brinkman_Twophase_sintetico_validacao2 import *

# from Brinkman_mono_sintetico import *
# from brinkman_mono_arapua import *

from Brinkman_Twophase_sintetico_IMPES_modificado_fulldomain import *


import os
import time

_folder_base = [
    "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/testeKvug_semPermvug",
    # "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/ImpesModificado_256",
    # "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/ImpesModificado_128",
    # "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/ImpesModificado_64",
    # "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/ImpesModificado_32",
    # "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/vugg_sintetico/five_spot",
]


for i in _folder_base:
    try:
        os.mkdir(i)
    except OSError as error:
        print(error)

Nx = 50
Ny = 50
mu_w = 1
mu_o = 1
perm_matriz = 100  # md
perm_vug = 9.86923e16
dt = 200
pin = 2
pout = 1
IMPES_Steps = [512, 256, 128, 64, 32]
# # comentarios


# def DataRecord(t, dt, dir1):
#     f = open(dir1 + "/results_times" + ".txt", "w")
#     string = "step" + "," + "time (s)" + ","
#     f.write(string)
#     f.write("\n")
#     for i in range(len(t)):
#         string = str(t[i]) + "," + str(float(dt[i])) + ","
#         f.write(string)
#         f.write("\n")
#     f.close()


# time_each_step = []


# for i in range(len(IMPES_Steps)):

#     start_time = time.time()
#     # BrinkmanIMPES(_folder_base[0], mu_w, mu_o, perm_matriz, dt)
#     BrinkmanIMPES(
#         Nx, _folder_base[i], mu_w, mu_o, perm_matriz, dt, pin, pout, IMPES_Steps[i]
#     )
#     final_time = time.time() - start_time

#     time_each_step.append(final_time)


# DataRecord(IMPES_Steps, time_each_step, _folder_base[0])


# BrinkmanIMPESGsmh(_folder_base[0], mu_w, mu_o, perm_matriz, dt)

# BrinkmanMonoBDM(perm_matriz, pin, pout, mu_w, Nx, Ny, _folder_base[0])
# BrinkmanIMPESGsmh(_folder_base[0], mu_w, mu_o, perm_matriz, dt)
BrinkmanIMPES(
    Nx,
    _folder_base[0],
    mu_w,
    mu_o,
    perm_matriz,
    perm_vug,
    dt,
    pin,
    pout,
    IMPES_Steps[4],
)
