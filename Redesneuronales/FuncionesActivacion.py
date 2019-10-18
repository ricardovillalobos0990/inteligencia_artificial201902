import numpy as np

class FuncionesActivacion:

    def sigmoide(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoide_derivado(self, x):
        return self.sigmoide(x) * (1 - self.sigmoide(x))

    def tangente(self, x):
        return np.tanh(x)

    def tangente_derivada(self, x):
        return 1 - x**2

    def relu(self, x):
        return 0 if x < 0  else 1