from tkinter import ttk
from tkinter import *
from Perceptron import Perceptron
from RedNeuronalSimple import RedNeuronalSimple
from RedNeuronalMulticapa import RedNeuronalMulticapa
from numpy import array, dot, random
import time

class RedesNeuronales:
    def __init__(self):
        self.ventana = Tk()
        self.ventana.title("Redes Neuronales")
        self.ventana.geometry("600x550")
        self.ventana.minsize(600, 550)
        self.ventana.configure(cursor="bogosity")
        self.notebook = ttk.Notebook(self.ventana)
        self.notebook.pack(fill='both',expand='yes')
        self.entradas = {}
        self.salidas = {}
        self.bahias = {}
        self.activacion = {}
        self.entrenamiento = {}
        self.instances = {}
        self.ob = {}
    
    def createFrame(self, instancia, title):
        """
        Recibe la instancion y el titulo de la ventana
        y devuelve la creacion de la pestaña creada con el titulo
        """
        self.instances[instancia] = ttk.Frame(self.notebook)
        self.notebook.add(self.instances.get(instancia), text=title)

    def interface(self, instancia, entradas, descripcion, Ob):
        """
        Recibe la instancia, el numero de entradas, una descripcion y el obejeto
        Ejecuta la interface con la instancia seleccionada, con el numero de entradas y salidas de la neurona
        """
        etiquetaEntradas = []
        etiquetaSalidas = []
        nombreEntradas = []
        nombreSalidas = []
        self.entradas[instancia] = []
        self.salidas[instancia] = []
        # Etiquetas en la instancia con recomendaciones de USO
        Label(self.instances.get(instancia), text="Ingrese la misma cantidad de item para cada entrada, separados por coma y sin espacio" + "\n"
        "como se indica en las guias de ejemplo de color VERDE ", fg='black' ).place(x=0 ,y=10)
        Label(self.instances.get(instancia), text=descripcion.get("nombre"), fg='green', font="Times 14 bold").place(x=5 ,y=60)
        Label(self.instances.get(instancia), text=descripcion.get("entradas"), fg='green', font="Times 14 bold").place(x=130 ,y=60)
        Label(self.instances.get(instancia), text=descripcion.get("salidas"), fg='green', font="Times 14 bold").place(x=380 ,y=60)
        # Genera las entradas y salidas segun el numero de entradas que recibe la interface y se itera segun ese numero generando los campos necesarios
        i = 0
        for i in range(entradas):
            etiquetaEntradas.append(Label(self.instances.get(instancia),text="Entrada # " + str(i+1) + ' :').place(x=10 ,y=100 + i * 20))
            self.entradas[instancia].append(StringVar())
            nombreEntradas.append(Entry(self.instances.get(instancia), textvariable=self.entradas[instancia][i]).place(x=110,y=100+ i * 20))

            self.salidas[instancia].append(StringVar())
            etiquetaSalidas.append(Label(self.instances.get(instancia),text="Salida # " + str(i+1) + ' :').place(x=300 ,y=100 + i * 20))
            nombreSalidas.append(Entry(self.instances.get(instancia), textvariable=self.salidas[instancia][i]).place(x=380,y=100 + i * 20))
        # Se declara la etiqueta dentron de la instanca, una descripcion, ubicacion y la variable que contiene el valor dentro de la Entrada
        Label(self.instances.get(instancia),text="Ingrese el sesgo de bahias").place(x=10 ,y=100 + (i+2) * 20)
        self.bahias[instancia] = DoubleVar()
        Entry(self.instances.get(instancia),textvariable=self.bahias[instancia]).place(x=200,y=100 + (i+2) * 20)

        Label(self.instances.get(instancia),text="Ingrese entrenamiento").place(x=10 ,y=100 + (i+3) * 20)
        self.entrenamiento[instancia] = IntVar()
        Entry(self.instances.get(instancia),textvariable=self.entrenamiento[instancia]).place(x=200,y=100 + (i+3) * 20)

        Label(self.instances.get(instancia),text="Seleccione la Funcion").place(x=10 ,y=100 + (i+4) * 20)
        self.activacion[instancia] = StringVar()
        ttk.Combobox(self.instances.get(instancia),values=("sigmoide","tangente"),textvariable=self.activacion[instancia]).place(x=200,y=100 + (i+4) * 20)

        Button(self.instances.get(instancia),text="Entrenar",fg="black", bg="white", font="fixedsys 14 bold", cursor="spider", command=lambda: self.callNeurona(instancia)).place(x=0 ,y=100 + (i+7) * 20, height=40, width=550)
        self.ob[instancia] = Ob

    def callNeurona(self, instancia):
        total = len(self.entradas[instancia])
        entradas = []
        salidas = []

        for i in range(total):
            entradas.append([float(x) for x in self.entradas[instancia][i].get().split(sep=',')])
            salida  = [float(x) for x in self.salidas[instancia][i].get().split(sep=',')]
            if len(salida) == 1: 
                salidas.append(float(salida[0]))
            else:
                 salidas.append(salida)

        ob = self.ob[instancia](entradas, salidas, self.bahias[instancia].get(), self.entrenamiento[instancia].get(), self.activacion[instancia].get())

        #Se asigna el valor de TIEMPO a la variable inicio, antes de la funcion entrenar
        inicio =time.time()
        ob.entrenar()
        final = time.time()
        tiempo = round( final - inicio )
        #Se asigna el valir de TIEMPO a la variable final, despues de la funcion entrenar

        cadena = ob.imprimirResultado()


        Label(self.instances.get(instancia), text="RESULTADO", fg="red").place(x=0 ,y=100 + (i+9) * 20)
        Label( self.instances.get(instancia), text=cadena, fg='red' ).place(x=0 ,y=100 + (i+10) * 20)
        Label( self.instances.get( instancia ), text="El tiempo que se tardo fue de: " + str(tiempo) + " Segundos", fg='red' ).place( x=0, y=100 + (i + 17) * 20 )

        ob.generarGrafico()

    def sostenerVentana(self):
        """
        No recibe Parametro
        Y ejecuta la ventana en el bucle principal
        """
        self.ventana.mainloop()

per = {'nombre':'Perceptron' , "entradas" : "0,0,1" , "salidas" : " 1 "  }
redsim = {'nombre':'Red Simple' , "entradas" : "1,1,1" , "salidas" : " 1 "  }
redmul = {'nombre':'Red Multiple' , "entradas" : "0,1" , "salidas" : " 1,0 "  }

ob = RedesNeuronales()
ob.createFrame('red', 'Perceptrón')
ob.createFrame('red2', 'Red neuronal simple')
ob.createFrame('red3', 'Red neuronal multicapa')
ob.interface('red', 4, per, Perceptron)
ob.interface('red2', 4, redsim, RedNeuronalSimple)
ob.interface('red3', 7, redmul, RedNeuronalMulticapa)
ob.sostenerVentana()
