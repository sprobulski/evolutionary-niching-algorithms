import math
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools
from deap.algorithms import varAnd
import random
import os

# Configuration class to store shared parameters
class Config:
    decode_range_min = -1  # Minimum decoding range value for x
    decode_range_max = 1   # Maximum decoding range value for x
    genotype_length = 16  # Genotype length (number of bits)
    mutation_bit_probability = 0.01  # Probability of mutating a single bit
    population_size = 500  # Population size
    CXPB = 0.8  # Crossover probability
    MUTPB = 0.01  # Mutation probability
    NGEN = 2000  # Number of generations
    threshold = 0.2  # Threshold for grouping extrema
    min_cluster_size = 10  # Minimum cluster size
    fitness_function = None  # Objective function
    optimization_type = "max"  # max or min
    niching_method = "crowding"  # "crowding", "sharing", "clearing", "speciation"
    sigma_share = 0.1  # For Fitness Sharing
    sigma_clear = 0.1  # For Clearing
    capacity = 1       # For Clearing
    sigma_species = 0.1  # For Speciation

# Configuration object
config = Config()

# Global variable for toolbox
toolbox = None

# Default multimodal objective function
def default_multimodal_function(x):
    return 2 * math.exp(-20 * (x + 0.7)**2) + 1.5 * math.exp(-20 * (x - 0.1)**2) + 3 * math.exp(-20 * (x - 0.7)**2)

# Multimodal objective function for individuals
def multimodal_function(individual, function=default_multimodal_function):
    x = decode_individual(individual)
    return (function(x),)

# Assign the default objective function in config
config.fitness_function = default_multimodal_function

# Function to set configuration parameters
def set_config(decode_range_min, decode_range_max, genotype_length, mutation_bit_probability, 
               population_size, CXPB, MUTPB, NGEN, threshold, min_cluster_size, 
               custom_fitness_function=None, optimization_type="max", niching_method="crowding",
               sigma_share=0.1, sigma_clear=0.1, capacity=1, sigma_species=0.1):
    config.decode_range_min = decode_range_min
    config.decode_range_max = decode_range_max
    config.genotype_length = genotype_length
    config.mutation_bit_probability = mutation_bit_probability
    config.population_size = population_size
    config.CXPB = CXPB
    config.MUTPB = MUTPB
    config.NGEN = NGEN
    config.threshold = threshold
    config.min_cluster_size = min_cluster_size
    if optimization_type not in ["max", "min"]:
        raise ValueError("optimization_type musi być 'max' lub 'min'")
    config.optimization_type = optimization_type
    if niching_method not in ["crowding", "sharing", "clearing", "speciation"]:
        raise ValueError("niching_method musi być 'crowding', 'sharing', 'clearing' lub 'speciation'")
    config.niching_method = niching_method
    config.sigma_share = sigma_share
    config.sigma_clear = sigma_clear
    config.capacity = capacity
    config.sigma_species = sigma_species
    if custom_fitness_function is not None:
        config.fitness_function = custom_fitness_function

# Function to return the current configuration as a dictionary
def get_config():
    return {
        'population_size': config.population_size,
        'CXPB': config.CXPB,
        'MUTPB': config.MUTPB,
        'NGEN': config.NGEN,
        'threshold': config.threshold,
        'min_cluster_size': config.min_cluster_size,
        'optimization_type': config.optimization_type,
        'niching_method': config.niching_method,
        'sigma_share': config.sigma_share,
        'sigma_clear': config.sigma_clear,
        'capacity': config.capacity,
        'sigma_species': config.sigma_species,
        'decode_range_min': config.decode_range_min,
        'decode_range_max': config.decode_range_max
    }

# Function to decode binary to decimal within a specified range
def decode_individual(individual):
    decimal = sum(bit * (2 ** i) for i, bit in enumerate(reversed(individual)))
    max_decimal = (2 ** config.genotype_length) - 1
    if max_decimal == 0:  
        return config.decode_range_min
    x = config.decode_range_min + (decimal / max_decimal) * (config.decode_range_max - config.decode_range_min)
    return x

# Compute distance between individuals in phenotype space
def distance(ind1, ind2):
    x1 = decode_individual(ind1)
    x2 = decode_individual(ind2)
    return abs(x1 - x2)


# Selection method used in crowding to preserve diversity
def crowding_selection(parents, offspring):
    config = get_config() 
    optimization_type = config['optimization_type']
    
    # Comparison function based on optimization direction
    if optimization_type == "max":
        compare = lambda child_fitness, parent_fitness: child_fitness > parent_fitness
    else:  # "min"
        compare = lambda child_fitness, parent_fitness: child_fitness < parent_fitness

    new_population = []
    for i in range(0, len(offspring), 2):
        if i + 1 < len(offspring):
            child1, child2 = offspring[i], offspring[i + 1]
            parent1, parent2 = parents[i], parents[i + 1]
            
            d_c1_p1 = distance(child1, parent1)
            d_c1_p2 = distance(child1, parent2)
            d_c2_p1 = distance(child2, parent1)
            d_c2_p2 = distance(child2, parent2)
            
            if d_c1_p1 + d_c2_p2 < d_c1_p2 + d_c2_p1:
                if compare(child1.fitness.values[0], parent1.fitness.values[0]):
                    new_population.append(child1)
                else:
                    new_population.append(parent1)
                if compare(child2.fitness.values[0], parent2.fitness.values[0]):
                    new_population.append(child2)
                else:
                    new_population.append(parent2)
            else:
                if compare(child1.fitness.values[0], parent2.fitness.values[0]):
                    new_population.append(child1)
                else:
                    new_population.append(parent2)
                if compare(child2.fitness.values[0], parent1.fitness.values[0]):
                    new_population.append(child2)
                else:
                    new_population.append(parent1)
        else:
            child = offspring[i]
            parent = parents[i]
            if compare(child.fitness.values[0], parent.fitness.values[0]):
                new_population.append(child)
            else:
                new_population.append(parent)
    return new_population

# Sharing function for fitness sharing
def sharing_function(distance, sigma_share):
    if distance < sigma_share:
        return 1 - (distance / sigma_share)**2
    return 0

# Implementation of fitness sharing for niching
def fitness_sharing(pop, sigma_share):
    for i, ind1 in enumerate(pop):
        sharing_sum = 0
        for ind2 in pop:
            dist = distance(ind1, ind2)
            sharing_sum += sharing_function(dist, sigma_share)
        
        if sharing_sum > 0:  
            original_fitness = ind1.fitness.values[0]
            ind1.fitness.values = (original_fitness / sharing_sum,)

def clearing(pop, sigma_clear, capacity):
    config = get_config()
    optimization_type = config['optimization_type']
    
    # Sort population by fitness
    pop.sort(key=lambda ind: ind.fitness.values[0], reverse=True if optimization_type == "max" else False)
    
    for i in range(len(pop)):
        if (pop[i].fitness.values[0] == 0 and optimization_type == "max") or \
           (pop[i].fitness.values[0] == float('inf') and optimization_type == "min"):
            continue
        
        winner = pop[i]
        count = 1  # Counter for individuals in the niche
        
        for j in range(i + 1, len(pop)):
            if (pop[j].fitness.values[0] == 0 and optimization_type == "max") or \
               (pop[j].fitness.values[0] == float('inf') and optimization_type == "min"):
                continue
            
            if distance(winner, pop[j]) < sigma_clear:
                if count < capacity:
                    count += 1
                else:
                    pop[j].fitness.values = (0,) if optimization_type == "max" else (float('inf'),)




def speciation(pop, sigma_species):
    config = get_config()
    optimization_type = config['optimization_type']
    
    # Sort population by fitness
    if optimization_type == "max":
        pop.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
    else:  # "min"
        pop.sort(key=lambda ind: ind.fitness.values[0])
    
    # Division of population into species
    species = []
    unassigned = list(pop)
    
    while unassigned:
        representative = unassigned[0]
        current_species = [representative]
        unassigned.remove(representative)
        
        i = 0
        while i < len(unassigned):
            ind = unassigned[i]
            if distance(ind, representative) < sigma_species:
                current_species.append(ind)
                unassigned.pop(i)
            else:
                i += 1
        
        species.append(current_species)
    
    return species

# Visualization of current population 
def plot_population(pop, gen, save=False):
    x_vals = [decode_individual(ind) for ind in pop]
    fitness_vals = [config.fitness_function(x) for x in x_vals]
    x = np.linspace(config.decode_range_min, config.decode_range_max, 1000)
    y = [config.fitness_function(xi) for xi in x]
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'b-', label='Fitness function')
    plt.scatter(x_vals, fitness_vals, c='red', s=10, label='Population')
    plt.title(f'Population in generation {gen}')
    plt.xlabel('x')
    plt.ylabel('Fitness value')
    plt.grid(True)
    plt.legend()
    
    if save:
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig(f'plots/pop_gen_{gen}.png')
    plt.show()
    plt.close()

# Visualization of fitness evolution across generations
def plot_fitness_scatter(pop_history, current_gen, save=False):
    plt.figure(figsize=(10, 6))
    
    config = get_config()
    optimization_type = config['optimization_type']
    
    global_max_fitness = float('-inf')
    global_min_fitness = float('inf')
    

    for gen in range(current_gen + 1):
        pop = pop_history[gen]
        fitness_vals = [ind.fitness.values[0] for ind in pop]
        if fitness_vals:  
            gen_max_fitness = max(fitness_vals)
            gen_min_fitness = min(fitness_vals)
            

            global_max_fitness = max(global_max_fitness, gen_max_fitness)
            global_min_fitness = min(global_min_fitness, gen_min_fitness)
    

    for gen in range(current_gen + 1):
        pop = pop_history[gen]
        fitness_vals = [ind.fitness.values[0] for ind in pop]
        gens = [gen] * len(fitness_vals)
        plt.scatter(gens, fitness_vals, c='black', s=3, alpha=0.3)


    plt.title(f'Evolvution of fitness value for generation {current_gen}')
    plt.xlabel('Generation')
    plt.ylabel('Fitness value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(-1, current_gen + 1)
    
    margin = (global_max_fitness - global_min_fitness) * 0.1  # 10% margin
    plt.ylim(global_min_fitness - margin, global_max_fitness + margin)
    
    plt.legend()
    if save:
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig(f'plots/scatter_gen_{current_gen}.png')
    plt.show()
    plt.close()

# Analysis of extrema in the final population
def find_ekstremum(pop, threshold=None, min_cluster_size=None):
    threshold = config.threshold if threshold is None else threshold
    min_cluster_size = config.min_cluster_size if min_cluster_size is None else min_cluster_size
    optimization_type = config.optimization_type  


    individuals = [(decode_individual(ind), ind.fitness.values[0], ind) for ind in pop]
    individuals.sort(key=lambda x: x[0])  
    

    clusters = []
    current_cluster = [individuals[0]]
    
    for i in range(1, len(individuals)):
        if abs(individuals[i][0] - current_cluster[-1][0]) < threshold:
            current_cluster.append(individuals[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [individuals[i]]
    if current_cluster:
        clusters.append(current_cluster)
    

    min_cluster_size = min_cluster_size if min_cluster_size > 0 else 1
    clusters = [cluster for cluster in clusters if len(cluster) >= min_cluster_size]
    

    

    extrema = []
    for cluster in clusters:

        avg_x = sum(ind[0] for ind in cluster) / len(cluster)
        avg_fitness = sum(ind[1] for ind in cluster) / len(cluster)
        

        if optimization_type == "max":
            best_ind = max(cluster, key=lambda ind: ind[1]) 
        else:  # "min"
            best_ind = min(cluster, key=lambda ind: ind[1]) 
        

        extrema.append((avg_x, avg_fitness, best_ind, cluster))
    

    if optimization_type == "max":
        extrema.sort(key=lambda x: x[1], reverse=True)  
    else:  # "min"
        extrema.sort(key=lambda x: x[1]) 

    num_extrema = len(extrema)
    extrema_values = [(avg_x, avg_fitness) for avg_x, avg_fitness, _, _ in extrema]
    clusters = [cluster for _, _, _, cluster in extrema]
    
    return num_extrema, extrema_values, clusters

# Function to configure DEAP
def configure_deap():
    global toolbox
    if config.optimization_type == "max":
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    else:  # "min"
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

    # Create toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bit", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bit, n=config.genotype_length)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=config.population_size)
    toolbox.register("evaluate", lambda ind: multimodal_function(ind, function=config.fitness_function))
    toolbox.register("mate", tools.cxOnePoint)  # One-point crossover
    toolbox.register("mutate", tools.mutFlipBit, indpb=config.mutation_bit_probability)
    toolbox.register("select", tools.selRoulette)

    return toolbox

# Function to initialize the population
def evolve_population(pop, toolbox, pop_history, plot_generations):
    config = get_config()
    niching_method = config['niching_method']
    cxpb = config['CXPB']
    mutpb = config['MUTPB']
    ngen = config['NGEN']
    
    # Evolve the population
    for gen in range(ngen):
        # Selection method based on niching
        if niching_method in ["clearing", "speciation"]:
            toolbox.register("select", tools.selRoulette)
        else:
            toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Keep a portion of the best individuals (elite)
        elite_size = 5 if niching_method != "crowding" else 0
        elite = tools.selBest(pop, elite_size)
        
        if niching_method == "clearing":
            sigma_clear = config['sigma_clear']
            capacity = config['capacity']
            clearing(pop, sigma_clear, capacity)
            
            # Choose parents and combine with elite
            parents = toolbox.select(pop, len(pop) - elite_size)
            offspring = list(elite) + parents[:len(pop) - elite_size]
            
            # Crossover and mutation
            offspring = varAnd(offspring, toolbox, cxpb, mutpb)
            
            # Calculate fitness for new individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            pop[:] = offspring
        
        elif niching_method == "sharing":
            sigma_share = config['sigma_share']
            fitness_sharing(pop, sigma_share)
            
            # Choose parents and combine with elite
            parents = toolbox.select(pop, len(pop) - elite_size)
            offspring = list(elite) + parents[:len(pop) - elite_size]
            
            # Crossover and mutation
            offspring = varAnd(offspring, toolbox, cxpb, mutpb)
            
            # Calculate fitness for new individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            pop[:] = offspring
        
        elif niching_method == "speciation":
            sigma_species = config['sigma_species']
            
            # Sort population by fitness
            if config['optimization_type'] == "max":
                pop.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
            else:  # "min"
                pop.sort(key=lambda ind: ind.fitness.values[0])
            
            # Division of population into species
            species = []
            unassigned = list(pop)
            
            while unassigned:
                representative = unassigned[0]
                current_species = [representative]
                unassigned.remove(representative)
                
                i = 0
                while i < len(unassigned):
                    ind = unassigned[i]
                    if distance(ind, representative) < sigma_species:
                        current_species.append(ind)
                        unassigned.pop(i)
                    else:
                        i += 1
                
                species.append(current_species)
            
            # Evolve each species
            offspring = []
            for spec in species:
                # Keep the best individual from the species
                offspring.append(spec[0])
                
                # Choose parents for crossover
                spec_pop = list(spec)
                if len(spec_pop) > 1:  # Crossover only if there are at least 2 individuals
                    spec_parents = toolbox.select(spec_pop, len(spec_pop) - 1)
                    spec_offspring = varAnd(spec_parents, toolbox, cxpb, mutpb)
                    
                    invalid_ind = [ind for ind in spec_offspring if not ind.fitness.valid]
                    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                    for ind, fit in zip(invalid_ind, fitnesses):
                        ind.fitness.values = fit
                    
                    offspring.extend(spec_offspring)
            
            # Combine offspring with elite
            offspring.extend(elite)
            
            while len(offspring) < len(pop):
                new_ind = toolbox.individual()
                fitness = toolbox.evaluate(new_ind)
                new_ind.fitness.values = fitness
                offspring.append(new_ind)
            
            offspring = offspring[:len(pop)]
            pop[:] = offspring
        
        elif niching_method == "crowding":
            # Randomly shuffle the population for crowding
            pop_copy = list(map(toolbox.clone, pop))  
            random.shuffle(pop_copy) 
            offspring = []
            
            #  Crossover and mutation
            for i in range(0, len(pop_copy), 2):
                if i + 1 < len(pop_copy) and random.random() < cxpb:  
                    # Select parents for crossover
                    parent1, parent2 = pop_copy[i], pop_copy[i + 1]
                    child1, child2 = map(toolbox.clone, [parent1, parent2])
                    # Crossover
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
                    offspring.extend([child1, child2])
                else:
                    # No crossover, just add parents to offspring
                    offspring.append(pop_copy[i])
                    if i + 1 < len(pop_copy):
                        offspring.append(pop_copy[i + 1])
            
            # Mutation
            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            

            fitnesses = list(map(toolbox.evaluate, offspring))
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit
            
            # Crowding selection
            new_pop = crowding_selection(pop_copy, offspring)
            

            if len(new_pop) < len(pop):
                new_pop.extend(pop_copy[len(new_pop):])
            elif len(new_pop) > len(pop):
                new_pop = new_pop[:len(pop)]
            
            pop[:] = new_pop
        
        else:
            # Default selection method
            parents = toolbox.select(pop, len(pop) - elite_size)
            offspring = list(elite) + parents[:len(pop) - elite_size]
            
            # Crossover and mutation
            offspring = varAnd(offspring, toolbox, cxpb, mutpb)
            

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            pop[:] = offspring
        

        pop_history.append(list(map(toolbox.clone, pop)))
        
        # Plotting the population and fitness evolution
        if gen + 1 in plot_generations:
            print(f"Generacja {gen + 1}:")
            plot_population(pop, gen + 1, save=False)
            plot_fitness_scatter(pop_history, gen + 1, save=False)
    
    return pop