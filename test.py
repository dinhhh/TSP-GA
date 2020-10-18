#from numpy import random
import numpy
import random

class City(object):
	"""docstring for ClassName"""
	def __init__(self, name, x, y):
		super(City, self).__init__()
		self.name = name
		self.x = x
		self.y = y
	def printInfo(self):
		print(str(self.name) + " (" + str(self.x) + ", " + str(self.y) + ")")
	def distanceTo(self, target):
		delX = self.x - target.x
		delY = self.y - target.y
		return numpy.sqrt(delX*delX + delY*delY)
class Individual(object):
	"""docstring for Individual"""
	def __init__(self, chromosome):
		super(Individual, self).__init__()
		self.chromosome = chromosome
		self.fitness = self.calFitness()
	def calFitness(self):
		sum = self.chromosome[len(self.chromosome) - 1].distanceTo(self.chromosome[0])
		for i in range(len(self.chromosome) - 1):
			sum += self.chromosome[i].distanceTo(self.chromosome[i+1])
		return 1 / sum
	def printInfo(self):
		for i in range (len(self.chromosome)):
			self.chromosome[i].printInfo()
		print("Fitness value = " + str(self.fitness))

def createChromosome(cityList, startCity):
	chromosome = []
	if startCity not in cityList:
		print("START CITY NOT EXIST")
	else:
		chromosome.append(startCity)
		cityList.remove(startCity)
		chromosome += random.sample(cityList, len(cityList))
		cityList.append(startCity)
	return chromosome

def printChromosome(chromosome):
	for i in range (len(chromosome)):
		chromosome[i].printInfo()

def initPopulation(cityList, sizeOfPopulation, startCity):
	population = []
	for i in range(sizeOfPopulation):
		population.append(Individual(createChromosome(cityList, startCity)))
	return population

def printPopulation(population):
	print("POPULATION INFORMATION:")
	for i in range(len(population)):
		population[i].printInfo()

def printFitnessPopulation(population):
	print("FITNESS INFORMATION:")
	for i in range (len(population)):
		print(population[i].fitness)

def selection(population, selectionSize, elitism):
	selectionResult = []
	sortedList = sorted(population, key= lambda x: x.fitness, reverse=True)
	for i in range (elitism):
		selectionResult.append(sortedList[i])
	del sortedList[:elitism]

	selectionResult += random.sample(sortedList, selectionSize - elitism)
	return selectionResult

def crossover(parent1, parent2):
	child = []
	childP1 = []
	childP2 = []

	geneA = int(random.random() * len(parent1.chromosome))
	geneB = int(random.random() * len(parent1.chromosome))
	
	startGene = min(geneA, geneB)
	endGene = max(geneA, geneB)
	for i in range(startGene, endGene):
		childP1.append(parent1.chromosome[i])
		
	childP2 = [item for item in parent2.chromosome if item not in childP1]

	child = childP1 + childP2
	return Individual(child)

def crossoverPopulation(population, selectionSize, elitism):
	selectionResult = selection(population, selectionSize, elitism)
	result = []
	for i in range (selectionSize):
		p1 = random.choice(selectionResult)
		p2 = random.choice(selectionResult)
		child = crossover(p1, p2)
		result.append(child)
	r = random.sample(population, len(population) - selectionSize)
	result += r
	return result

def mutate(individual, mutateRate):
	for swapped in range(len(individual.chromosome)):
		if(random.random() < mutateRate):
			swapWith = int(random.random() * len(individual.chromosome))
			
			city1 = individual.chromosome[swapped]
			city2 = individual.chromosome[swapWith]
			
			individual.chromosome[swapped] = city2
			individual.chromosome[swapWith] = city1
	return individual

def mutatedPopulation(population, mutateRate):
	result = []
	for i in range(len(population)):
		m = mutate(population[i], mutateRate)
		result.append(m)
	return result

def nextGeneration(population, selectionSize, elitism, mutateRate):
	crossoverPopu = crossoverPopulation(population, selectionSize, elitism)
	nextGene = mutatedPopulation(crossoverPopulation, mutateRate)
	return nextGene

def algorithm(population, selectionSize, elitism, mutateRate, loops):
	popu = population
	for i in range (loops):
		popu = nextGeneration(popu, selectionSize, elitism, mutateRate)
		print("LOOPS TIME: " + str(i))
	return popu

def main():
	cityList = []
	for i in range(1, 50):
		cityList.append(City(i, i*10, i*20))
	
	popu = initPopulation(cityList, 10, cityList[0])
	printFitnessPopulation(popu)
	solution = algorithm(popu, 5, 2, 0.1, 500)
	#mutatedPopulation(popu, 0.01)



if __name__ == '__main__':
	main()
		