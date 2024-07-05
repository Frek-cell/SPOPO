import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from IPython.display import display # type: ignore


class Omegas:
    def __init__(self, center: float = 0, edge: float = 3, points: int = 100):
        self.createMesh(center, edge, points)

    def createMesh(self, center, edge, points):
        omegas = np.linspace(center, edge, points // 2)
        omegas_inv = list(omegas - (edge - center))
        omegas_inv.extend(x for x in omegas if x not in omegas_inv)
        self.grid = omegas_inv


class Line:
    def __init__(self, name, mesh: Omegas = None, detuning: float = 0.1, bypassTime: float = 1.0,
                 spectralWidth: float = 1.0,
                 phase: float = np.pi / 2, mode: int = 0, tuneCnm: float = 0.2, tuneCnn: float = 1.0,
                 tuneAmpl: float = 0.2, tuneDecay: float = 1.0, tuneAmplMode: str = 'exp', tuneDecayMode: str = 'exp'):

        self.X = []
        self.Y = []

        kwargs = {'detuning': detuning,
                  'bypassTime': bypassTime,
                  'spectralWidth': spectralWidth,
                  'phase': phase,
                  'mode': mode,
                  'tuneCnm': tuneCnm,
                  'tuneCnn': tuneCnn,
                  'tuneAmpl': tuneAmpl,
                  'tuneDecay': tuneDecay,
                  'tuneAmplMode': tuneAmplMode,
                  'tuneDecayMode': tuneDecayMode}
        self.tuneY(mesh, **kwargs)

        self.df = pd.DataFrame(kwargs, index=(name,))

    def tuneY(self, mesh: Omegas = None, **kwargs):
        # Setup parameters
        self.d = kwargs.get('detuning')
        self.tr = kwargs.get('bypassTime')
        self.s = kwargs.get('spectralWidth')

        # Measurment parameters
        self.phi = kwargs.get('phase')
        self.n = kwargs.get('mode')

        # Tuning parameters
        self.cnm = kwargs.get('tuneCnm')
        self.cnn = kwargs.get('tuneCnn')
        self.l = kwargs.get('tuneAmpl')
        self.g = kwargs.get('tuneDecay')
        self.l_mode = kwargs.get('tuneAmplMode')
        self.g_mode = kwargs.get('tuneDecayMode')

        if mesh is not None:
            self.changeX(mesh)
        else:
            self._getY()

    def changeX(self, mesh):
        self.X = mesh.grid
        self._getY()

    def _getY(self):
        Y = []
        for x in self.X:
            Y.append(Dot(self).I(x))

        self.Y = Y


class Dot:
    def __init__(self, line: Line):
        # Setup parameters
        self.d = line.d
        self.tr = line.tr
        self.s = line.s

        # Measurment parameters
        self.phi = line.phi
        self.n = line.n

        # Tuning parameters
        self.cnm = line.cnm
        self.cnn = line.cnn
        self.l = line.l
        self.g = line.g
        self.l_mode = line.l_mode
        self.g_mode = line.g_mode

    def _n(self, key):
        if key is None:
            return self.n

        else:
            return key

    def _delta(self, mode=None):
        n = self._n(mode)

        return self.d - self.cnn * self.C(n, n)

    def _gamma(self, mode=None):
        n = self._n(mode)

        if self.g_mode == 'exp':
            return self.g / np.exp(-0.1 * pow(n, 2))

        elif self.g_mode == 'flat':
            return self.g

        elif self.g_mode == 'free':
            return 0

    def _lambda(self, mode=None):
        n = self._n(mode)

        if self.l_mode == 'exp':
            return self.l * pow(0.9, n)

        elif self.l_mode == 'spec1':
            if n == 0:
                return self.l
            return 0

        elif self.l_mode == 'spec2':
            if n in (0, 2):
                return self.l
            return 0
        
        elif self.l_mode == 'spec5':
            if n in (0, 1, 2, 3, 4):
                return self.l
            return 0

        elif self.l_mode == 'flat':
            return self.l

        elif self.l_mode == 'free':
            return 0

    def C(self, n, m):
        if n is None:
            n = self.n

        if m == n and m > -1:
            fraction = (-(2 * n + 1) / (2 * pow(self.s, 2)))
            return self.cnm * fraction

        elif n > 1 and m == n - 2:
            fraction = (np.sqrt((n - 1) * n) / (2 * pow(self.s, 2)))
            return self.cnm * fraction

        elif m == n + 2:
            fraction = (np.sqrt((n + 1) * (n + 2)) / (2 * pow(self.s, 2)))
            return self.cnm * fraction

        return 0

    def _H(self, x, mode=None):
        term1 = self._gamma(mode) / 2 - 1j * (self._delta(mode) - x)
        term2 = pow(self._lambda(mode) / 2, 2) / (self._gamma(mode) / 2 + 1j * (self._delta(mode) + x))

        return term1 - term2

    def _U(self, x, mode=None):
        return 1 / self._H(x, mode=mode)

    def _V(self, x, mode=None):
        term1 = self._U(x, mode=mode)
        term2 = (self._lambda(mode) / 2) / (self._gamma(mode) / 2 + 1j * (self._delta(mode) + x))

        return term1 * term2

    def _W(self, x, mode=None):
        return self._U(x, mode=mode) - 1 / self._gamma(mode)

    def _u(self, x, m, n=None):
        factor = 1j * self.C(n, m)
        term1 = self._V(x, mode=n) * self._V(-x, mode=m).conjugate()
        term2 = self._U(x, mode=n) * self._U(x, mode=m)

        return factor * (term1 - term2)

    def _v(self, x, m, n=None):
        factor = 1j * self.C(n, m)
        term1 = self._V(x, mode=n) * self._U(-x, mode=m).conjugate()
        term2 = self._U(x, mode=n) * self._V(x, mode=m)

        return factor * (term1 - term2)

    def _a(self, x, p, m, n=None):
        factor = 1j * self.C(m, p)
        term1 = self._u(x, m, n=n) * self._U(x, mode=p)
        term2 = self._v(x, m, n=n) * self._V(-x, mode=p).conjugate()

        return factor * (term2 - term1)

    def _b(self, x, p, m, n=None):
        factor = 1j * self.C(m, p)
        term1 = self._u(x, m, n=n) * self._V(x, mode=p)
        term2 = self._v(x, m, n=n) * self._U(-x, mode=p).conjugate()

        return factor * (term2 - term1)

    def _I0(self, x):
        term1 = self._gamma() * self._W(x) * self._W(x).conjugate()
        term2 = self._gamma() * self._V(-x) * self._V(-x).conjugate()
        vw = self._gamma() * self._W(x) * self._V(-x)
        factor = np.exp(1j * 2 * self.phi)
        term3 = factor * vw

        I0 = term1 + term2 + term3 + term3.conjugate()

        return I0.real * self._gamma() * self.tr

    def _I2(self, x):
        alphas, betas = 0, 0
        us, vs, uvs = 0, 0, 0

        for m in [self.n - 2, self.n + 2]:
            alphas += self._a(x, self.n, m).conjugate()
            betas += self._b(-x, self.n, m)
            us += self._u(x, m) * self._u(x, m).conjugate() * self._gamma(mode=m)
            vs += self._v(-x, m) * self._v(-x, m).conjugate() * self._gamma(mode=m)
            uvs += self._u(x, m) * self._v(-x, m) * self._gamma(mode=m)

        term1 = self._gamma() * self._W(x) * alphas
        term2 = self._gamma() * self._V(-x).conjugate() * betas
        term3 = self._gamma() * self._V(-x).conjugate() * alphas * np.exp(-1j * 2 * self.phi)
        term4 = self._gamma() * self._W(x) * betas * np.exp(1j * 2 * self.phi)
        term5 = uvs * np.exp(1j * 2 * self.phi)

        halfI = term1 + term2 + term3 + term4 + term5
        I2 = halfI + halfI.conjugate() + us + vs

        return I2.real * self._gamma() * self.tr

    def I(self, x):
        return self._I0(x) + self._I2(x)


class Data:
    def __init__(self, X: Omegas, lines: list[Line], n_as_index=True):

        indx = X.grid
        #         names =

        if n_as_index:
            cols = [f'n = {line.n}' for line in lines]
        else:
            cols = [line.df.index.to_list()[0] for line in lines]

        dt = [line.Y for line in lines]
        data = {name: data_row for name, data_row in zip(cols, dt)}

        self.df = pd.DataFrame(data=data, index=indx)
        self.info = pd.concat([line.df for line in lines], axis=0)

        self.info.index = pd.Series(cols)

    def plot(self, size=(12, 7), show_info=False, save=False, name='Default', title=None, fontsize=18):
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams.update({'font.size': fontsize})
        plt.rc('legend', fontsize=14)

        fig, ax = plt.subplots(figsize=size)
        plt.rcParams.update({'font.size': fontsize})
        plt.rc('legend', fontsize=14)

        ax.set_xlabel('$\Omega/\gamma$')
        ax.set_ylabel('$< I^\dagger I >$')
        plt.rcParams.update({'font.size': fontsize})
        plt.rc('legend', fontsize=14)

        X = self.df.index
        listY = self.df.values.transpose().tolist()

        ax.plot(X, np.ones(len(X)), '--', color='black', label='vacuum')
        plt.rcParams.update({'font.size': fontsize})
        plt.rc('legend', fontsize=14)

        for k in range(len(listY)):
            plt.rcParams.update({'font.size': fontsize})
            plt.rc('legend', fontsize=14)
            ax.plot(X, listY[k], label=f'{self.df.columns[k]}')

        ax.legend()
        ax.grid()

        if title:
            plt.title(title)

        if save:
            plt.savefig(f'{name}.pdf')

        plt.show()

        if show_info:
            display(self.info)


    def plotX(self, size=(12, 7), show_info=False, save=False, name='Default', title=None, fontsize=18, style=None):
        fig, ax = plt.subplots(figsize=size)

        ax.set_xlabel('$\Omega/\gamma$')
        ax.set_ylabel('$< I^\dagger I >$')

        X = self.df.index
        listY = self.df.values.transpose().tolist()

        for k in range(len(listY)):
            if k in (0, 1) and style:
                ax.plot(X, 10*np.log10(listY[k]), label=f'{self.df.columns[k]}', linestyle=style)
            else:
                ax.plot(X, 10*np.log10(listY[k]), label=f'{self.df.columns[k]}')

        ax.legend(loc='upper right')
        ax.grid()

        if title:
            plt.title(title)

        if save:
            plt.savefig(f'{name}.pdf')

        plt.show()

        if show_info:
            display(self.info)