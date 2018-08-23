# coding: utf-8

import numpy as np


def ReLU(z):
    return np.max((0,z))  #Definición de la función ReLU para un elemento

ReLU = np.vectorize(ReLU) #Modificando la función ReLU para que reciba como argumento arreglos


#Bloque para ingreso de opciones de usuario
inicio = int(input("Eljia entre las dos opciones 1 para matrices de unos y 0 para matrices aleatorias: "))
X=np.array(list(map(float,input("Ingrese datos separados por espacios: ").split(" ")))).reshape([1,-1])
ncapas = int(input("Ingrese el número de capas: " ))

neurona=np.zeros(ncapas+1) #Inicializando vector para almacenar número de neuronas por capa
neurona[0] = int(X.shape[-1]) #Ingresando el número de atributos del archivo de entrada 

for i in range (ncapas):
    neurona[i+1] = int(input('Ingrese el número de neuronas para capa {}: '.format(i+1))) #Asignando a cada capa el número de neuronas deseadas


if inicio == 0: #Bloque para para matrices de unos

    for i in range (1,ncapas+1):
        A=np.random.rand(int(neurona[i-1]),int(neurona[i])) #Definiendo matriz A para cada capa
        B=np.random.rand(1,int(neurona[i])) #Definienndo matriz B para cada capa
        X=ReLU(np.dot(X,A) + B) #Calculando X para cada capa

elif inicio == 1:

    for i in range (1,ncapas+1):
        A=np.ones((int(neurona[i-1]),int(neurona[i]))) #Definiendo matriz A para cada capa
        B=np.ones((1,int(neurona[i]))) #Definienndo matriz B para cada capa
        X=ReLU(np.dot(X,A) + B) #Calculando X para cada capa

print(X) #Retronando la X final

