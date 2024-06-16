# from Brinkman_Twophase_sintetico_deltaP import *

from Brinkman import *

# from Brinkman_Twophase_sintetico_validacao2 import *

# from Brinkman_mono_sintetico import *
# from brinkman_mono_arapua import *

import os
import time

_folder_base = [
    "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/Impes",
    # "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/ImpesModificado_256",
    # "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/ImpesModificado_128",
    # "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/ImpesModificado_64",
    # "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/ImpesModificado_32",
    # "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/ImpesModificado_16",
    # "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/ImpesModificado_8",
    # "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/ImpesModificado_4",
    # "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/ImpesModificado_2",
    # "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/ImpesModificado_1",
    # "/home/tfk/Desktop/results/Brinkman/Brinkman_Biphase/vugg_sintetico/five_spot",
]


for i in _folder_base:
    try:
        os.mkdir(i)
    except OSError as error:
        print(error)

Nx = 50
Ny = 50
mu_w = 0.001
mu_o = 0.001
perm_matriz = 100  # md
dt = 200
pin = 2
pout = 1
IMPES_Steps = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
# comentarios


def DataRecord(t, dt, dir1):
    f = open(dir1 + "/results_times" + ".txt", "w")
    string = "step" + "," + "time (s)" + ","
    f.write(string)
    f.write("\n")
    for i in range(len(t)):
        string = str(t[i]) + "," + str(float(dt[i])) + ","
        f.write(string)
        f.write("\n")
    f.close()


time_each_step = []


# for i in range(len(IMPES_Steps)):

#     start_time = time.time()
#     # BrinkmanIMPES(_folder_base[0], mu_w, mu_o, perm_matriz, dt)
#     BrinkmanIMPES(
#         Nx, _folder_base[i], mu_w, mu_o, perm_matriz, dt, pin, pout, IMPES_Steps[i]
#     )
#     final_time = time.time() - start_time

#     time_each_step.append(final_time)


# DataRecord(IMPES_Steps, time_each_step, _folder_base[0])

BrinkmanIMPES(
    Nx, _folder_base[0], mu_w, mu_o, perm_matriz, dt, pin, pout, IMPES_Steps[5]
)

# BrinkmanIMPESGsmh(_folder_base[0], mu_w, mu_o, perm_matriz, dt)

# BrinkmanMonoBDM(perm_matriz, pin, pout, mu_w, Nx, Ny, _folder_base[0])
# BrinkmanIMPESGsmh(_folder_base[0], mu_w, mu_o, perm_matriz, dt)
# BrinkmanIMPES(Nx, _folder_base[0], mu_w, mu_o, perm_matriz, dt, pin, pout)
