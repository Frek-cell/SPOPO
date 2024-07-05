import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from bibl import Line, Data, Omegas
from itertools import product

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 18})
plt.rc('legend', fontsize=14)

N = 0

C = 0.1
l0 = 1/3
ampl_mode = 'spec1'

for N in (0, 2):
    for Cl in [(0.25, 1/3),]:
        for ampl_mode in ('spec1', 'spec2'):
        # for ampl_mode in ('spec5', 'exp'):
            title = 'bigfontFCP_'
            C, l0 = Cl

            if C == 0 and l0 == 0.1715:
                title += 'NODISP_'
            elif (C, l0) == (0.1, 0.1715):
                title += 'I'
            elif (C, l0) == (0.1, 1/3):
                title += 'II'
            elif (C, l0) == (0.25, 1/3):
                title += 'III'
            elif (C, l0) == (0.25, 0.477592):
                title += 'IV'
            else:
                title += 'UNKNOWN'

            if ampl_mode == 'spec1':
                title += 'A'
            elif ampl_mode == 'spec2':
                title += 'B'
            elif ampl_mode == 'exp':
                title += 'C'
            elif ampl_mode == 'spec5':
                title += 'D'

            title += str(N)
            title += '.pdf'

            omegas = Omegas(points=300)

            lines = []
            for phi in np.arange(0, 3.14, 0.01):
                lines.append(
                    Line(
                        name = phi,
                        mesh = omegas,
                        mode = N,
                        phase = phi,
                        detuning = 0.0,
                        tuneCnm = C,
                        tuneAmpl = l0,
                        tuneDecay = 1.0,
                        tuneAmplMode = ampl_mode,
                        tuneDecayMode = 'flat'
                    )
                )

            graph = Data(omegas, lines, n_as_index=False)
            df = 10*np.log10(graph.df)

            def get_squeezing(x, y):
                return df[y].loc[x]


            v = np.vectorize(get_squeezing)
            x = df.index.to_list()
            y = df.columns.to_list()

            X, Y = np.meshgrid(x, y)
            Z = v(X, Y)

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 7))

            CP = ax.pcolor(X, Y, Z, cmap=cm.Spectral_r, vmin=-5, vmax=5)  #
            CP2 = ax.contour(X, Y, Z, colors='black', levels=9)
            plt.clabel(CP2, inline = True, fontsize=30, fmt='%1.1f', colors = 'black')

            ax.set_title(f'$N = {N}$')
            ax.set_xlabel('frequency, $\Omega/\gamma$')
            ax.set_ylabel('LO phase, rad')

            cbar = fig.colorbar(CP)
            cbar.ax.set_ylabel('squeezing, dB')

            # cbar.add_lines(CP2)
            fig.savefig(title)

            # plt.show()