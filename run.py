import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bibl import Line, Data, Omegas

modes = (4,)
C = 0.25
l0 = 0.3333
graph_name = 'P_BIII_z'

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 18})
plt.rc('legend', fontsize=14)

ampl_mode = 'spec2'
omegas = Omegas(points=300)
lines = []
for N in modes:
    for phi in (0, np.pi/2):
        line = Line(
            name=f'$\phi = {round(phi, 2)}$, $n = {N}$',
            mesh=omegas,
            mode=N,
            phase=phi,
            detuning=0.0,
            tuneCnm=C,
            tuneAmpl=l0,
            tuneDecay=1.0,
            tuneAmplMode=ampl_mode,
            tuneDecayMode='flat'
        )

        lines.append(line)

graph = Data(omegas, lines, n_as_index=False)

graph_title = f'$C/\gamma = {C}, \lambda_0/\gamma = {l0}$'

graph.plotX(
    size=(12, 7),
    name = graph_name,
    title = graph_title,
    save = True
)
