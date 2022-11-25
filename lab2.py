import random
import math
import numpy as np
import time

f = open(r"instancias\cap41.txt", "r")
resultado = open("resultados.txt", "w")

line = f.readline() 
data = line.split()
num_loc = int(data[0]) #numero de locaciones
num_cust = int(data[1]) #numero de clientes
loc = list(range(num_loc))
cust = list(range(num_cust))
cap = [] #capacidad
fc = [] #costo
dem = [] #demanda
vc = [] #costo 
for i in loc:
    line = f.readline() 
    data = line.split()
    cap.append(int(data[0]))
    fc.append(float(data[1]))
for i in cust:
    line = f.readline()
    dem.append(int(line))
    j = 0
    vc.append([])
    while j < num_loc:
        line = f.readline()
        data = line.split()
        for aux in data:
            vc[i].append(float(aux))
            j += 1
f.close()

#print(cap) #vector de capacidades
#print(fc)  #vector de costos de facilities
#print(dem) #vector de demanda de clientes
#print(vc)  #Matriz de costo de solventar al cliente i desde la fabrica j

params = {}

params["capacidades"] = cap
params["costoFac"] = fc
params["demanda"] = dem
params["costoCliente"] = vc


#############################################Binarización##########################
def binariza(matrix):
    matrixProbT = 0.5*(np.power(np.abs(matrix),1.5))

    matrixRand = np.random.uniform(low=0.0,high=1.0,size=matrix.shape)
    matrixBinOut = np.greater(matrixProbT,matrixRand).astype(int)

    return matrixBinOut

#############################Reparación de soluciones ultra simple################
def repara(vector,cap_min,cap_general):
    for i in range(len(vector)):
        if vector[i] == 0:
            vector[i] == 1
            if cumple(vector,cap_min,cap_general) == 1:
                return vector

#################################Verificación#####################################
#si el vector es una solución posible
def cumple(vector,cap_min, cap_facilities):
    cap_general = 0
    for i in range(len(vector)):
        cap_general += cap_facilities[i] * vector[i]

    if cap_general > cap_min:
        return 1
    else:
        return 0

###############################FITNESS##############################################
def obtenerFitness(poblacion,matrix,solutionsRanking,params):

    cap_min = sum(params['demanda']) #capacidad general mínima para cubrir a los clientes
    cap_facilities = params["capacidades"]
    costo_facilities = params["costoFac"]

    matrix = binariza(poblacion)
    print(matrix.shape[0])
    for solucion in range(matrix.shape[0]):
        if cumple(matrix[solucion],cap_min, cap_facilities) == 0:
            aux = repara(matrix[solucion],cap_min,cap_facilities)
            print(type(solucion),solucion)
            matrix[solucion] = aux

    #Calculamos Fitness
    fitness = np.sum(np.multiply(matrix,costo_facilities),axis =1)
    solutionsRanking = np.argsort(fitness) # rankings de los mejores fitness

    return matrix,fitness,solutionsRanking

####################################################################################



#########################################################################Seno coseno
def iterarSCA(maxIter, t, dimension, poblacion, bestSolutionCon):
    # nuestro valor de a es constante y es extraido desde paper
    a = 2
    # calculamos nuestro r1
    r1 = a - (t * (a / maxIter))
    # for de individuos
    for i in range(poblacion.__len__()):
        # for de dimensiones
        for j in range(dimension):
            # calculo un numero aleatoreo entre [0,1]
            rand = random.uniform(0.0, 1.0)
            # calculo r2
            r2 =  (2 * math.pi) * rand 
            # calculo r3
            r3 = 2 * rand
            # calculo r4
            r4 = random.uniform(0.0, 1.0)
            if r4 < 0.5:
                # perturbo la poblacion utilizando como base la funcion seno
                poblacion[i][j] = poblacion[i][j] + ( ( ( r1 * math.sin(r2)) * abs( ( r3 * bestSolutionCon[j] ) - poblacion[i][j] ) ) )
            else:
                # perturbo la poblacion utilizando como base la funcion coseno
                poblacion[i][j] = poblacion[i][j] + ( ( ( r1 * math.cos(r2)) * abs( ( r3 * bestSolutionCon[j] ) - poblacion[i][j] ) ) )
    # retorno la poblacion modificada 
    return np.array(poblacion)
####################################################################################


dim = len(fc) #dimensiones del problema
pob = 40      #tamaño de la población
maxIter = 10

#inicializo poblacion continua
poblacion = np.random.uniform(low=-10.0, high=10.0, size=(pob,dim))

# Genero una población inicial binaria, esto ya que nuestro problema es binario (cflp)
matrixBin = np.random.randint(low=0, high=2, size = (pob,dim))

# Genero un vector donde almacenaré los fitness de cada individuo
fitness = np.zeros(pob)

# Genero un vetor dedonde tendré mis soluciones rankeadas
solutionsRanking = np.zeros(pob)

# Calculo mi fitness inicial
matrixBin,fitness,solutionsRanking = obtenerFitness(poblacion,matrixBin,solutionsRanking,params)

for iter in range(0, maxIter):
    
    # obtengo mi tiempo inicial
    processTime = time.process_time()  
    timerStart = time.time()
    
    # DETERMINO MI MEJOR SOLUCION Y LA GUARDO 
    bestRowAux = solutionsRanking[0]
    Best = poblacion[bestRowAux]
    BestBinary = matrixBin[bestRowAux]
    BestFitness = np.min(fitness)
    
    # perturbo la poblacion con la metaheuristica, pueden usar SCA y GWO
    # en las funciones internas tenemos los otros dos for, for de individuos y for de dimensiones


    poblacion = iterarSCA(maxIter, iter, dim, poblacion.tolist(), Best.tolist())

    
    #Binarizamos y evaluamos el fitness de todas las soluciones de la iteración t
    matrixBin,fitness,solutionsRanking = obtenerFitness(poblacion,matrixBin,solutionsRanking,params)

    #Conservo el Best
    if fitness[bestRowAux] > BestFitness:
        fitness[bestRowAux] = BestFitness
        matrixBin[bestRowAux] = BestBinary
    BestFitnes = np.min(fitness)

    # Obtengo parametro de diversidad, SI TIENEN MAS DUDAS DE ESTO PUEDEN HABLARME AL CORREO 
    #diversidades, maxDiversidad, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(poblacion,maxDiversidad)

    timerFinal = time.time()
    # calculo mi tiempo para la iteracion t
    timeEjecuted = timerFinal - timerStart
    print("iteracion: "+str(iter)+", best fitness: "+str(np.min(fitness))+", tiempo iteracion (s): "+str(timeEjecuted))
    resultado.write("iteracion: "+str(iter)+", best fitness: "+str(np.min(fitness))+", tiempo iteracion (s): "+str(timeEjecuted)+"\n")

print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("Best fitness: "+str(BestFitnes))
resultado.write("Best fitness: "+str(BestFitnes)+"\n")
print("Cantidad de columnas seleccionadas: "+str(sum(BestBinary)))
resultado.write("Cantidad de columnas seleccionadas: "+str(sum(BestBinary))+"\n")
print("Best solucion: \n"+str(BestBinary.tolist()))
resultado.write("Best solucion: \n"+str(BestBinary.tolist())+"\n")
print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

