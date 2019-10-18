from numpy import exp, array, random, dot, ravel, sum, abs
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
            self.errores.append(abs(sum(error)))
            ajuste = dot(self.entradas.T, error * self.activacion_prima(salida))
            self.pesos_signaticos += ajuste
            self.esperados.append(ajuste)
            
    def pensar(self, entrada):
        return self.activacion(dot(entrada, self.pesos_signaticos))


    def imprimirResultado(self):
        entrada_prueba = array([1,0,0])
        return f"Predicion para la entrada {entrada_prueba} es {self.pensar(entrada_prueba)}"

    def validaractivacion(self):
        if self.nombreactivacion == 'sigmoide':
            self.activacion = self.sigmoide
            self.activacion_prima = self.sigmoide_derivado
        elif self.nombreactivacion == 'tangente':
            self.activacion = self.tangente
            self.activacion_prima = self.tangente_derivada

    def generarGrafico(self):
        plt.title("Perceptron")
        plt.plot(self.errores, '-',color='red', Label="Errores")
        plt.legend(loc="upper right")
        plt.show()