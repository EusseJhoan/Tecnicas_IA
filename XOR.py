import numpy as np


def Sigmoid(z):
    return 1/(1+np.exp(-z)) #Definición función sigma

Sigmoid = np.vectorize(Sigmoid) #Vectorización función sigma para trabajar con arreglos


def Dsigmoid(z):
    return (1/(1+np.exp(-z)))*(1-(1/(1+np.exp(-z))))  # Definición derivada de función sigma 

Dsigmoid = np.vectorize(Dsigmoid) #Vectorización derivada de función sigma para trabajar con arreglos


X=np.array([[0,0],[0,1],[1,0],[1,1]]) #Entrada
Y=np.array([[0],[1],[1],[0]]) # Valores reales para y
N=int(X.shape[0]) #Número de datos
ncapas=2 #Numero de capas
neucapa0 = int(X.shape[-1]) #Numero características
neucapa1 = 2 #Numero de neuronas capa 1
neucapa2 = 1 #Numero de neuronas capa 2
neurona=[neucapa0,neucapa1,neucapa2] #Aregglo para almacenar numero de neuronas por capa
alpha=10 #Hiperparámetro del SGD (lo puse en 10 pues en valores mas bajos se me atascaba en 0.5)



A=[0,] #Arreglo para almacenar pesos
B=[0,] #Arreglo para almacenar interceptos
Z=[0,] #Arreglo para almacenar predcciones sin función de activación
Xi=[X,] #Arreglo para almacenar predcciones sin función de activación


#Inicialización
for i in range (1,ncapas+1): 
	A.append(np.random.uniform(-1,1,(int(neurona[i-1]),int(neurona[i]))))
	B.append(np.random.uniform(-1,1,(1,int(neurona[i]))))
	Z.append(np.dot(Xi[i-1],A[i]) + B[i])
	Xi.append(Sigmoid(Z[i]))
	
    
deltaf= 2*(Xi[-1]-Y)/N
delta=[0,deltaf,]
Delta=[0,]
GradA=[0,]
GradB=[0,]


for j in range(5000): #5000 épocas

    
	for k in range(1,ncapas+1): #Backpropagation
    
		Delta.append(delta[k]*Dsigmoid(Z[-k]))
		GradA.append(np.dot(Xi[-(k+1)].T,Delta[k]))
		GradB.append(Delta[k].sum(axis=0))
		delta.append(np.dot(Delta[k],A[-k].T))
        
	for l in range(1,ncapas+1): #Actualización pesos
        
		A[l]= A[l] - alpha*GradA[-l]
		B[l]= B[l] - alpha*GradB[-l]
    
    
	Z=[0,]
	Xi=[X,]
    
	for m in range (1,ncapas+1): #Inferencia
        
		Z.append(np.dot(Xi[m-1],A[m]) + B[m])
		Xi.append(Sigmoid(Z[m]))
	
	deltaf= 2*(Xi[-1]-Y)/N
	
	delta=[0,deltaf,]
	Delta=[0,]
	GradA=[0,]
	GradB=[0,]
        
            
print(Xi[-1]) #Predicción

