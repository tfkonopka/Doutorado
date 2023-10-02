import matplotlib.pyplot as plt
import numpy as np


_str1 = [
    "Arapua10_mesh1.txt",
    "Arapua10_mesh2.txt",
    "Arapua10_mesh3.txt",
    "Arapua10_mesh4.txt",
    "Arapua10_mesh017.txt",
    "Arapua10_results.txt",
]

_str2 = [
    "Arapua10_mesh1",
    "Arapua10_mesh2",
    "Arapua10_mesh3",
    "Arapua10_mesh4",
    "Arapua10_mesh017",
    "Arapua10_results",
]

# _str1 = [
#     "Arapua17_mesh1.txt",
#     "Arapua17_mesh2.txt",
#     "Arapua17_mesh3.txt",
#     "Arapua17_mesh4.txt",
#     # "Arapua17_results.txt",
# ]

# _str2 = [
#     "Arapua17_mesh1",
#     "Arapua17_mesh2",
#     "Arapua17_mesh3",
#     "Arapua17_mesh4",
#     # "Arapua17_results",
# ]

# _str1 = [
#     "Arapua24_mesh1.txt",
#     "Arapua24_mesh2.txt",
#     "Arapua24_mesh3.txt",
#     "Arapua24_mesh4.txt",
#     "Arapua24_results.txt",
# ]


# _str2 = [
#     "Arapua24_mesh1",
#     "Arapua24_mesh2",
#     "Arapua24_mesh3",
#     "Arapua24_mesh4",
#     "Arapua24_results",
# ]


# _str1 = [
#     "vug4_mesh20.txt",
#     "vug4_mesh40.txt",
#     "vug4_mesh80.txt",
#     "vug4_mesh100.txt",
#     "vug4_mesh120.txt",
# ]

# _str2 = [
#     "mesh_vug4_20",
#     "mesh_vug4_40",
#     "mesh_vug4_80",
#     "mesh_vug4_100",
#     "mesh_vug4_120",
# ]

# _str1 = [
#     "mesh_20.txt",
#     "mesh_40.txt",
#     "mesh_80.txt",
#     "mesh_100.txt",
#     "mesh_120.txt",
# ]

# _str2 = [
#     "mesh_20",
#     "mesh_40",
#     "mesh_80",
#     "mesh_100",
#     "mesh_120",
# ]

# _str2 = ["mesh1", "mesh2", "mesh3", "mesh4"]

_caminho = "/home/tfk/Desktop/results/Validation/figuras_tese/Arapua/Arapua_Mesh_test/"

color = ["b-", "r", "g", "c", "m", "y"]

# phi = 0.328  # sintetico
phi = 1.089  # Arapua10
# phi = 1.581  # Arapua17
# phi = 1.396  # Arapua24


def DataRecord(td, NpD, pin, _filename):
    f = open(+"saida" * _filename + ".txt", "w")
    string = "td" + "    " + "NpD"
    f.write(string)
    f.write("\n")
    for i in range(len(td)):
        string = str(td[i]) + "    " + str(float(NpD[i])) + "    " + str(float(pin[i]))
        f.write(string)
        f.write("\n")
    f.close()


def DataRecord_tD_Npd(td, Npd, _filename, _path):
    f = open(_path + "/" + "saida" + "/" + _filename + ".txt", "w")
    string = "td" + "    " + "Qw"
    f.write(string)
    f.write("\n")
    for i in range(len(td)):
        if i % 40 == 0:
            string = str(td[i]) + "    " + str(float(Npd[i]))
            f.write(string)
            f.write("\n")
    f.close()


def DataRecord_tD_Bsw(td, Bsw, _filename, _path):
    f = open(_path + "/" + "saida" + "/" + _filename + ".txt", "w")
    string = "td" + "    " + "Bsw"
    f.write(string)
    f.write("\n")
    for i in range(len(td)):
        if i % 50 == 0:
            string = (
                "{:.5f}".format((float(td[i])))
                + "    "
                + "{:.5f}".format((float(Bsw[i])))
            )
            f.write(string)
            f.write("\n")
    f.close()


def le_dados(_str1):
    ifs = open(_str1, "r")
    data = []

    data = ifs.readlines()
    ifs.close()

    time = []
    Qo = []
    Qw = []

    for i in range(len(data)):
        if i == 0:
            continue
        temp = data[i].split(",")
        time.append(float(temp[0]))
        Qo.append(float(temp[2]))
        Qw.append(float(temp[3]))

    return time, Qo, Qw


def td_Npd(time, Qo, Qw):
    td = []
    time = np.array(time)
    td.append(0)
    Q = []
    Q.append(0)
    Npd = []
    Npd.append(0)

    for i in range(len(time) - 1):
        dt = time[i + 1] - time[i]
        Qm = (Qo[i + 1] + Qo[i]) / 2
        Qmedio = Qm * dt
        Q.append(Qmedio)

    print(f"Q_len = {len(Q)}")
    print(f"Qo = {len(Qo)}")

    td = 1 / phi * time * Qo[0]
    # td = time * Qo[0]

    for i in range(1, len(Q)):
        y = Npd[i - 1] + Q[i]
        Npd.append(y)

    Npd = np.array(Npd) / phi
    # Npd = np.array(Npd)

    return td, Npd


for j in range(len(_str1)):
    Bsw1 = []
    time1, Qo1, Qw1 = le_dados(_caminho + _str1[j])
    for i in range(len(Qo1)):
        Bsw1.append(Qw1[i] / (Qw1[i] + Qo1[i]))

    td1, Npdc1 = td_Npd(time1, Qo1, Qw1)
    DataRecord_tD_Bsw(td1, Bsw1, _str2[j], _caminho)
    # DataRecord_tD_Npd(td1, Npdc1, _str2[j], _caminho)

    plt.figure(1, figsize=(8, 4))
    plt.plot(td1, Npdc1, color[j], label=_str2[j], linewidth=1)

    plt.figure(2, figsize=(8, 4))
    plt.plot(td1, Bsw1, color[j], label=_str2[j], linewidth=1)


# plt.title("Arapua17", fontsize=12)
plt.legend(loc="lower right", shadow=True, fontsize=8)
# plt.minorticks_on()
# plt.xlim(0, 2)
# plt.ylim(0, 1)
# plt.xticks(fontsize=9)
# plt.yticks(fontsize=9)

# # plt.grid(b=True, which="major", color="gray", linestyle="--")
# # plt.grid(b=True, which="minor", color="lightgray", linestyle="dotted")

# plt.xlabel("$t_{D} $", fontsize=10)
# plt.ylabel("$Np_{D} $", fontsize=10)
# # plt.ylabel("Qo and Qw $ \\frac{m^{3}}{s}$ ")
# # plt.ylabel("Npd", fontsize=8)
# plt.savefig("Sem_Ajuste", bbox_inches="tight")

plt.show()
