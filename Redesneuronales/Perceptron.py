from random import choice
from numpy import array, dot, random
import matplotlib.pyplot as plt
from FuncionesActivacion import FuncionesActivacion

class Perceptron(FuncionesActivacion):
    def __init__(self, entradas, salidas, bahias, n, activacion="relu"):
        total = len(entradas)
        entrenamiento = []
        for i in range(total):
            entrenamiento.append(array((array(entradas[i]), salidas[i])))
    
        self.entrenamiento = entrenamiento
        self.w = random.rand(3)
        self.errores = []
        self.esperados = []
        self.bahias = bahias
        self.n = n

    def entrenar(self):
        for i in range(self.n):
            x, esperado = choice(self.entrenamiento)
            resultado = dot(self.w, x)
            self.esperados.append(esperado)
            error = esperado - self.relu(resultado)
            self.errores.append(error)
            #Ajuste
            self.w += self.bahias * error * x

    def imprimirResultado(self):
        cadena = ""
        for x, _ in self.entrenamiento:
            resultado = dot(self.w, x)
            cadena += f"{x[:3]}: {resultado} -> {self.relu(resultado)} \n"
        return cadena

    def generarGrafico(self):
        plt.title("Perceptron")
        plt.plot(self.errores,'-',color='red', Label="Errores")
        plt.plot(self.esperados,'*', color='green', Label="Esperados")
        plt.legend(loc="lower right")
        plt.show()