import random
import math
import numpy as np
import time
from amplpy import AMPL, Environment
from sys import *
import os

resultado = open("resultados.txt", "w")

#___________________LECTURA____________________________
#archivo = r"instancias\cap131.txt"
archivo = r"instancias\capa.txt"

f = open(archivo, "r")

print(argv[0])

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

capacidad_ab = 8000

for i in loc:
    line = f.readline()
    data = line.split()

    if i%25 == 0 and i != 0:
        capacidad_ab = capacidad_ab + 2000

    if archivo == r"instancias\capa.txt" or archivo == r"instancias\capb.txt":
        cap.append(capacidad_ab)
    else:
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

params["num_cust"] = num_cust
params["num_loc"] = num_loc
params["costoFac"] = fc
params["capacidades"] = cap
params["demanda"] = dem
params["costoCliente"] = vc


#___________________METAHEURÍSTICA____________________________


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
            vector[i] = 1
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
    #print(matrix)
    for solucion in range(matrix.shape[0]):
        #print("cumple =",cumple(matrix[solucion],cap_min, cap_facilities))
        if cumple(matrix[solucion],cap_min, cap_facilities) == 0: #se verifica que cumpla con las restricciones
            aux = repara(matrix[solucion],cap_min,cap_facilities) #si no lo hace, se repara la solución
            matrix[solucion] = aux

    #Calculamos Fitness
    fitness = np.sum(np.multiply(matrix,costo_facilities),axis =1)
    solutionsRanking = np.argsort(fitness) # rankings de los mejores fitness

    return matrix,fitness,solutionsRanking

####################################################################################



######################ALGORITMO SENO COSENO########################################
def iterarSCA(maxIter, t, dimension, poblacion, bestSolutionCon):
    #Constante sacada de paper
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
    BestFitness = np.min(fitness)

    # Obtengo parametro de diversidad, SI TIENEN MAS DUDAS DE ESTO PUEDEN HABLARME AL CORREO 
    #diversidades, maxDiversidad, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(poblacion,maxDiversidad)

    timerFinal = time.time()
    # calculo mi tiempo para la iteracion t
    timeEjecuted = timerFinal - timerStart
    print("iteracion: "+str(iter)+", best fitness: "+str(np.min(fitness))+", tiempo iteracion (s): "+str(timeEjecuted))
    resultado.write("iteracion: "+str(iter)+", best fitness: "+str(np.min(fitness))+", tiempo iteracion (s): "+str(timeEjecuted)+"\n")

print("Resultado final:",str(BestBinary.tolist()))
#instancia de ampl
#REQUIERE SER CAMBIADO PARA ESTA INSTANCIA, COMO TAMBIE MODIFICAR EL .MOD PARA ACEPTAR EL VECTOR DE FÁBRICAS COMO PARÁMETRO

rutaAMPL = r'D:\ProgramasWindows\ampl_mswin64'
#rutaAMPL = r'C:\Users\vicen\Desktop\10 Semestre\IOA\ampl_mswin64'

ampl = AMPL(Environment(rutaAMPL))

#model_directory = argv[2] if len(argv) == 3 else os.path.join("..", "IOA-CFLP-main")
model_directory = os.getcwd()
ampl.read(r"1_CFLP_model.mod")

#creacion del archivo.dat
file = "1_CFLP_data.dat"
with open(file,'w') as salida:
    salida.write("param loc := "+str(params["num_loc"])+';'+'\n') #fila de localidades
    salida.write("param cli := "+str(params["num_cust"])+';'+"\n") #fila de clientes

    salida.write("param facilities := ")
    for i in range(len(BestBinary)): #fila costo facilities
        salida.write(" "+str(i+1)+" ")
        salida.write(str(BestBinary[i]))
    salida.write(";"+'\n')

    salida.write("param FC := ")
    for i in range(len(params["costoFac"])): #fila costo facilities
        salida.write(" "+str(i+1)+" ")
        salida.write(str(params["costoFac"][i]))
    salida.write(";"+'\n')

    salida.write("param ICap := ")
    for i in range(len(params["capacidades"])): #fila capacidades facilities
        salida.write(" "+str(i+1)+" ")
        salida.write(str(params["capacidades"][i]))
    salida.write(";"+'\n')    

    salida.write("param dem := ")
    for i in range(len(params["demanda"])): #fila demanda clientes
        salida.write(" "+str(i+1)+" ")
        salida.write(str(params["demanda"][i]))
    salida.write(";"+'\n')

    salida.write("param TC :")
    for i in range(num_loc): # primera linea con las columnas del 
        salida.write(" "+str(i+1)+" ")
    salida.write(":=")    
    salida.write("\n")
    for i in range(num_cust):
        salida.write(str(i+1)+" ")
        for j in range(num_loc):
            salida.write(str(params["costoCliente"][i][j])+" ")
        salida.write("\n")
    salida.write("\n")


#pass data to ampl
ampl.read_data(file)
# Solve the model
ampl.solve()

print()
# Print out the result
print(
    "Objective function value: {}".format(ampl.get_objective("Total_Cost").value())
)

# Get the values of the variable Buy in a dataframe
#results = ampl.get_variable("y").get_values()
# Print
#print(results)
