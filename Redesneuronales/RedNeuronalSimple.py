from numpy import exp, array, random, dot, ravel
from FuncionesActivacion import FuncionesActivacion
import matplotlib.pyplot as plt

class RedNeuronalSimple(FuncionesActivacion):
    def __init__(self, entradas, salidas, bahias, n, activacion):
        self.nombreactivacion = activacion
        self.activacion = None
        self.activacion_prima = None
        self.pesos_signaticos = 2 * random.random((3,1)) - 1
        self.entradas = array(entradas)
        self.errores = []
        self.esperados= []
        self.salidas = array([salidas]).T
        self.bahias = bahias
        self.n = n
    
    def entrenar(self):
        self.validaractivacion()
        for i in range(self.n):
            salida = self.pensar(self.entradas)
            error = self.salidas - salida
            ajuste = dot(self.entradas.T, error * self.activacion_prima(salida))
            self.pesos_signaticos += ajuste
            self.esperados.append(ajuste)
            
    def pensar(self, entrada):
        return self.activacion(dot(entrada, self.pesos_signaticos))

    """
    def imprimirResultado(self):
        return "Hola"
    """

    def validaractivacion(self):
        if self.nombreactivacion == 'sigmoide':
            self.activacion = self.sigmoide
            self.activacion_prima = self.sigmoide_derivado
        elif self.nombreactivacion == 'tangente':
            self.activacion = self.tangente
            self.activacion_prima = self.tangente_derivada

    def generarGrafico(self):
        plt.title("Perceptron")
        plt.plot([1,2,3,4,5], '-',color='red', Label="Errores")
        plt.plot([9,8,7,6,5], '*', color='green', Label="Esperados")
        plt.legend(loc="upper right")
        plt.show()