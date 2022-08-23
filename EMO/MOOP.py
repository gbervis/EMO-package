from random import Random
from time import time
import matplotlib.pyplot as plt
import numpy as np
import math
import random as r

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# For combinations in Brokhoff & Zitzler
from itertools import combinations
import numpy as np

###################### OBJECTIVE REDUCTION - BROKHOFF & ZITZLER #################################
def MOSS_Exact(population, functions, n = None):
    if len(population[0]) != len(functions):
        return print("Error, size of objectives is not equal to coordinates size of individuals")
    
    # To take some solutions randomly
    if n != None:
        # Number of solutions in the population
        number_of_solutions = np.arange(len(population))
        # Number of solutions to take
        n = 5
        # Random index to take at the population
        random_indexs = np.random.choice(number_of_solutions, size=n, replace=False)
        # Solutions taken randomly from the population
        solutions = population[random_indexs, :]
    
    # To take all the solutions
    else:
        solutions = population
    
    ###### Begin of the algorithm
    final_list = []
    
    sets = performExactAlgorithm(solutions)
    for i in sets:
        ind = list(i)
        aux_list = [functions[j] for j in ind]
        final_list.append(aux_list)
    return final_list

def performExactAlgorithm(population):
    
    possibleSets = [set()]
    
    pairs = np.array(list(combinations(population,2)))
    for pair in pairs:
        currentSet = computeSetOfSetsFor(pair[0], pair[1])
        if len(currentSet) != set():
            union_ExactAlgo_2sets(possibleSets, currentSet)
    #return possibleSets
    return getSmallest(possibleSets) #possibleSets

# Obtiene una lista con los conjuntos más pequeños
def getSmallest(Set):
    if len(Set) == 0:
        return Set
    
    else:
        minimum = np.min([len(i) for i in Set])

        List = []
        for set_i in Set:
            if len(set_i) == minimum:
                List.append(set_i)
        return List
    
def computeSetOfSetsFor(p, q):
    sois = set()
    
    pSet = []
    qSet = []
    
    for i in range(len(p)):
        if p[i] <= q[i] and not q[i] <= p[i]: #Se ha quitado el =
            pSet.append({i})
        
        if q[i] <= p[i] and not p[i] <= q[i]:  #Se ha quitado el =
            qSet.append({i})
    
    
    if len(pSet) > 0 and len(qSet) > 0:
        #Esta union está en SetOfIntSet
        union_ExactAlgo_2sets(pSet, qSet)
        sois = pSet
    else:
        if len(pSet) == 0 and len(qSet) == 0:
            for i in range(len(p)):
                sois.add(i)
            
        
        if len(pSet) > 0:
            sois = pSet
            
        if len(qSet) > 0:
            sois = qSet
            
    return sois


# De la unión, ambos deben ser listas de objetivos
def union_ExactAlgo_2sets(Set1, Set2):
    newList = []
    
    # Decide quién de los dos es el más pequeño y el más grande
    if len(Set1) > len(Set2):
        smallset = Set2
        largeset = Set1
    else:
        smallset = Set1
        largeset = Set2
    
    if len(smallset) == 0:
        Set1.clear()
        Set1.append(largeset)#extend(largeset)
    
    else:
        for i in largeset:
            set1 = i
            for j in smallset:
                set2 = j
                unionset = set1.union(set2)
                addIfSubSet(unionset, newList)
        Set1.clear()
        for i in newList:
            Set1.append(i)

# Is must be a set, and List must be a list of sets
def addIfSubSet(Is, List):
    # insert = True iff Is has to be inserten into List    
    insert = True
    
    # a list of elements, that we have to delete in List
    delet = []
    
    for currentIs in List:
        if currentIs.issuperset(Is):
            delet.append(currentIs)
            insert = True

        else:
            if Is.issuperset(currentIs):
                insert = False
    if insert:
        for i in delet:
            List.remove(i)
        List.append(Is)

def Compute_Delta_Pop(population):
    pairs = np.array(list(combinations(population,2)))
    delta = 0
    for solution in pairs:
        if((solution[0] <= solution[1]).all()):
            delta = np.maximum(delta, np.max(solution[1] - solution[0]))
            

        elif((solution[0] >= solution[1]).all()):
            delta = np.maximum(delta, np.max(solution[0] - solution[1]))
            
    return delta

def Sets_Delta(pairs1,functions):
    ind = range(len(functions))
    for j in range(len(functions),0,-1):
        for i in combinations(ind,j):
            print(Compute_Delta_Pop(pairs1[:,i]), [functions[k] for k in i])

###################### OBJECTIVE REDUCTION - DEB & SAXENA #################################
def PCAs(population): 
    # Standaraization data
    pop_Std = StandardScaler().fit_transform(population).T#StandardScaler().fit_transform(population.T)

    # Getting the correlation matrix 
    R = np.corrcoef(pop_Std)
    # Calculating R*R^T
    RR_T = np.dot(R,R.T)

    # Getting the eigenvalues and eigenvectors
    eigen_values, eigen_vectors = np.linalg.eig(RR_T)

    # Sorting the eigenvector by decreasing order of the eigen values
    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:,idx]

    # Getting the contribution of every PC (decreasing order)
    variance_explained = eigen_values/np.sum(eigen_values)*100

    # Getting the cumulative variance
    cum_var = np.cumsum(variance_explained)

    return eigen_values, eigen_vectors, cum_var, R[:,idx]

def Obj_NonRed(population, functions, TC):
    eigen_values, eigen_vectors, cum_var, _ = PCAs(population)

    no_redundantObjectives = set()
    # Making calculus over the eigenvectors
    for i in range(len(cum_var)):
        v = eigen_vectors.T[i]

        # For second and so on components
        if i != 0:
            # Counting positive numbers
            positive_numbers = np.sum(np.where(v>0,1,0))

            if eigen_values[i] < 0.1:
                no_redundantObjectives.add(functions[np.where(v == np.max(np.abs(v)))[0][0]])#.__name__)

            elif eigen_values[i] >= 0.1 and cum_var[i] < TC:

                 # If all values are positives, take the most positive objective
                if positive_numbers == len(functions):
                    no_redundantObjectives.add(functions[np.where(v == np.max(v))[0][0]])#.__name__)

                # If all values are negatives, take all the objectives
                elif positive_numbers == 0:
                    for i in len(functions):
                        no_redundantObjectives.add(functions[i])#.__name__)

                else:
                    p, n = np.max(v), np.min(v)

                    # if p < |n|
                    if p < np.abs(n):
                        # if p >= 0.9|n|
                        if p >= 0.9*np.abs(n):

                            # we choose both objetives
                            no_redundantObjectives.add(functions[np.where(v == p)[0][0]])#.__name__)
                            no_redundantObjectives.add(functions[np.where(v == n)[0][0]])#.__name__)

                        # if p < 0.9|n| -> use n
                        else:
                            no_redundantObjectives.add(functions[np.where(v == n)[0][0]])#.__name__)

                    # if p > |n| 
                    elif p > np.abs(n):
                        # if p >= 0.8|n|
                        if p >= 0.8*np.abs(n):
                            # we choose both objetives
                            no_redundantObjectives.add(functions[np.where(v == p)[0][0]])#.__name__)
                            no_redundantObjectives.add(functions[np.where(v == n)[0][0]])#.__name__)

                        # if p < 0.8|n|
                        else:
                            no_redundantObjectives.add(functions[np.where(v == p)[0][0]])#.__name__)


        # For the first component
        #if i == 0:
        else:
            # Counting positive numbers
            positive_numbers = np.sum(np.where(v>0,1,0))

            # If all values are positives: 
            if positive_numbers == len(functions):
                no_redundantObjectives.add(functions[np.where(v == np.max(v))[0][0]])#.__name__)

            # If all values are negatives: 
            elif positive_numbers == 0:
                no_redundantObjectives.add(functions[np.where(v == np.min(v))[0][0]])#.__name__)

            # If there are positive and negative numbers
            else:
                p, n = np.max(v), np.min(v)
                no_redundantObjectives.add(functions[np.where(v == p)[0][0]])#.__name__)
                no_redundantObjectives.add(functions[np.where(v == n)[0][0]])#.__name__)


        if cum_var[i] < TC:
            continue
        else:
            break

    return no_redundantObjectives


#Llamando a los elementos de inspired
import inspired
from inspired import ec
from inspired.ec import terminators

class MOOP(object):    
    def __init__(self, functions, num_variables, intervals, vector = False, **args):
        seed = args.setdefault('seed', int(time()))#semilla = int(time())):
        rand = Random()
        rand.seed(seed)
        #Los métodos de estos paquetes serán de la clase MOOP
        #ec.emo.NSGA2.__init__(self,rand)
        #ec.emo.PAES.__init__(self,rand)
        
        #Definiendo los parámetros de esta clase
        self.functions = functions
        self.nvariables = num_variables
        self.seed = seed
        self.intervals = intervals
        self.eval = []
        
        global vtr
        vtr = vector
        
    # Función para la evaluación de los pares de funciones
    '''Este método bien puede graficar una o dos matrices'''
    @staticmethod
    def plotfun(m1,m2 = []):
        for i in range(len(m1)-1):
            for j in range(i+1,len(m1)):
                plt.title("Espacio de los objetivos")
                plt.xlabel(f"f{i+1}(x)")
                plt.ylabel(f"f{j+1}(x)")
                '''plt.plot(m1[i],m1[j],'.')
                if m2 == []:
                    plt.show()
                else:
                    plt.plot(m2[i],m2[j],'.',color='red')
                    plt.show()'''
                if m2 == []:
                    plt.plot(m1[i],m1[j],'.',color='red')
                    plt.show()
                else:
                    plt.plot(m1[i],m1[j],'.')
                    plt.plot(m2[i],m2[j],'.',color='red')
                    plt.show()
    
                    
    #Para reordenar por evaluación
    '''Será de utilidad para reordenar la población generada por el paquete
       inspyred, ya que como este lo da no es de mucha utilidad para graficar'''
    @staticmethod
    def reord(x):
        evaluation = []
        for i in range(len(x[0])):
            evaluation.append([j[i] for j in x])
        return evaluation
    
    #Evaluación de funciones
    '''Este método evalúa una lista en cada una de las funciones.
       Toma en cuenta, para esto, si se trata de una lista o una lista de listas.'''
    @staticmethod
    def evalfun(x, obj = [], *args):
        if (isinstance(x[0], list) or isinstance(x[0], np.ndarray)) == True:
            evaluation = []
            for f in obj:
                evaluation.append([f(x[i]) for i in range(len(x))])
            return evaluation
        # Si se tiene que la función regresa un vector
        elif vtr == True:
            return obj[0](x)
        
        else:
            evaluation = [f(x) for f in obj]
            return evaluation
        
        
    #Generador aleatorio, pensar en la posibilidad de añadir otro generador
    # que genere valores según los intervalos del Bounder
    def generador(self, random, args):
        size = args.get('num_inputs', 10)
        if isinstance(self.intervals[0],int) == True:  
            return [random.uniform(self.intervals[0], self.intervals[1]) for i in range(size)]
        else:
            return [random.uniform(self.intervals[0][i], self.intervals[1][i]) for i in range(size)]

    #Este es el evaluador que usará el paquete inspyred para realizar las evaluaciones
    def evaluador(self, candidates, args):
        functions = self.functions
        fitness = []
        for x in candidates:
            fitness.append(ec.emo.Pareto(self.evalfun(x,functions)))
        return fitness
    
    #Generando el espacio de objetivos
    #def EObj(self, self.nvariables)
        # Si la cantidad de variables es menor o igual a 3, generar el grid
        
        # Si es mayor, generar el espacio objetivo después de la evaluación del 
        
    def solvePAES(self, population_size = 100, maximize = False, 
                   num_generations = 100, mutation_rate = 0.25, plot_iter = False, objective_space = [],
                   iteration_step = 1, greater_than = 0, num_grid = 1, archive_name = None):
        rand = Random()
        rand.seed(self.seed)
        es = ec.emo.PAES(rand)
        #es.variator = [ec.variators.blend_crossover, 
                       #ec.variators.gaussian_mutation]
        es.terminator = terminators.generation_termination
        
        ### Para la verificación de intervalos
        if isinstance(self.intervals[0],int) == True:
            interv = ec.Bounder(self.intervals[0],self.intervals[1])
        else:
            interv = ec.Bounder(self.intervals)
            
        final_pop = es.evolve(generator = self.generador,
                              evaluator = self.evaluador,
                              pop_size = population_size,
                              maximize = maximize,
                              bounder = interv,
                              max_generations = num_generations,
                              mutation_rate = mutation_rate,
                              num_inputs = self.nvariables,
                              ##### Para la gráfica #######
                              plot_iter = plot_iter,
                              obj_space = objective_space,
                              step = iteration_step,
                              greater_than =  greater_than,
                              arch_name = archive_name,
                              a_name = 'PAES',
                              #############################
                              num_grid_divisions = num_grid)
        self.population = [final_pop[i].candidate for i in range(len(final_pop))]
        evalf = self.reord([final_pop[i].fitness for i in range(len(final_pop))])
        if np.sum(np.where(np.array(np.shape(evalf)) > 0,1,0)) >2:
            self.eval = np.array(evalf).T[0]
        else:
            self.eval = np.array(evalf).T
    
    def solveNSGA2(self, population_size = 100, maximize = False, 
                   num_generations = 100, mutation_rate = 0.25, plot_iter = False, objective_space=[],
                   iteration_step = 1, greater_than = 0, archive_name = None, stime = False):
        rand = Random()
        rand.seed(self.seed)
        es = inspired.ec.emo.NSGA2(rand)
        es.variator = [inspired.ec.variators.blend_crossover,
                       inspired.ec.variators.gaussian_mutation]
        es.terminator = terminators.generation_termination
        
        ### Para la verificación de intervalos
        if isinstance(self.intervals[0],int) == True:
            interv = ec.Bounder(self.intervals[0],self.intervals[1])
        else:
            interv = ec.Bounder(self.intervals)

        final_pop = es.evolve(generator = self.generador,
                              evaluator = self.evaluador,
                              pop_size = population_size,
                              maximize = maximize,
                              bounder = interv,
                              max_generations = num_generations,
                              mutation_rate = mutation_rate,
                              num_inputs = self.nvariables,
                              ##### Para la gráfica #######
                              plot_iter = plot_iter,
                              obj_space = objective_space,
                              step = iteration_step,
                              greater_than =  greater_than,
                              arch_name = archive_name,
                              a_name = 'NSGA-II',
                              #############################
                              functions = self.functions
                              )
        self.population = [final_pop[i].candidate for i in range(len(final_pop))]
        evalf = self.reord([final_pop[i].fitness for i in range(len(final_pop))])
        if np.sum(np.where(np.array(np.shape(evalf)) > 0,1,0)) >2:
            self.eval = np.array(evalf).T[0]
        else:
            self.eval = np.array(evalf).T
        #return self.plotfun(evalf)
    
    def solveMOPSO(self, population_size = 100, maximize = False, 
                   num_generations = 100, mutation_rate = 0.25, plot_iter = False, objective_space=[],
                   iteration_step = 1, greater_than = 0, neighborhoods = 3, inertia = 0.1, cognitive = 0.2, social = 0.15,archive_name = None):
        rand = Random()
        rand.seed(self.seed)
        es = inspired.swarm.PSO(rand)
        es.topology = inspired.swarm.topologies.ring_topology                                                     
        es.terminator = terminators.generation_termination
        
        ### Para la verificación de intervalos
        if isinstance(self.intervals[0],int) == True:
            interv = ec.Bounder(self.intervals[0],self.intervals[1])
        else:
            interv = ec.Bounder(self.intervals)
            
        final_pop = es.evolve(generator = self.generador,
                              evaluator = self.evaluador,
                              pop_size = population_size,
                              maximize = maximize,
                              bounder = interv,
                              max_generations = num_generations,
                              mutation_rate = mutation_rate,
                              num_inputs = self.nvariables,
                              ##### Para la gráfica #######
                              plot_iter = plot_iter,
                              obj_space = objective_space,
                              step = iteration_step,
                              greater_than =  greater_than,
                              arch_name = archive_name,
                              a_name = 'MOPSO',
                              #############################
                              neighborhood_size = neighborhoods,
                              inertia = inertia,
                              cognitive_rate = cognitive,
                              social_rate = social)
                              
        
        self.population = [final_pop[i].candidate for i in range(len(final_pop))]
        evalf = self.reord([final_pop[i].fitness for i in range(len(final_pop))])
        if np.sum(np.where(np.array(np.shape(evalf)) > 0,1,0)) >2:
            self.eval = np.array(evalf).T[0]
        else:
            self.eval = np.array(evalf).T
