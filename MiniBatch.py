import numpy as np

# # Importación de los datos de MultiLinReg # #

datos= np.genfromtxt('MultiLinReg.csv', delimiter= ',') # Extrayendo datos del archivo csv con Numpy
y=datos[:,-1] # Extrayendo los datos correspondientes a y 
y=y.reshape((-1,1)) # Reacomodandolas
X=datos[:,:-1] # Extrayendo los datos corrrespondientes a x
X=np.c_[np.ones((X.shape[0],1)),X] # Agregando vector fila a la Matriz X
A=np.random.rand(X.shape[1],1) #Generando Matriz de ai aleatorios
M=np.zeros((X.shape[1],1)) # Inicializando matriz de momentos con ceros
X.shape


# # Mini Batch con momentos codigo de uso general ##

alpha = 0.1 # definicion de hiper-parametro
beta = 0.9 # definicion de hiper-parametro
batchsize = 20 # Definición de el batchsize acorde a los datos que se tienen
datosbatch = int(X.shape[0]/batchsize) # Numero de datos por cada lote

for i in range(1000): # Ciclo para epocas

	#Revolviendo la Matriz X
	combined = list(zip(X, y)) 
	np.random.shuffle(combined)
	X[:], y[:] = zip(*combined)
    
	for j in range(1,int(batchsize)+1): # Ciclo para lotes 
        	Xl = X[datosbatch*(j-1):datosbatch*j,::] #Generando lotes de matriz X
        	Grad =(2/X.shape[0])*((np.dot(np.dot(Xl.T,Xl),A) - np.dot(Xl.T,y[datosbatch*(j-1):datosbatch*j]))) #Hallandoo el gradiente 
        	M = M*beta + alpha * Grad #Vector de momento
        	A = A - M #Vector fila de ai     

print(A)




