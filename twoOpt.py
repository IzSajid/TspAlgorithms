#!/usr/bin/env python
# coding: utf-8

# In[10]:


import random

class TravelingSalesmanTwoOpt:
    def __init__(self, cities):
        self.cities = cities
        self.num_cities = len(cities)
        self.tour = random.sample(cities, self.num_cities)

    def get_distance(self, city1, city2):
        x1, y1 = city1
        x2, y2 = city2
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def improve_tour(self):
        improved = True
        while improved:
            improved = False
            for i in range(self.num_cities - 1):
                for j in range(i + 2, self.num_cities):
                    if self.get_distance(self.tour[i], self.tour[i + 1]) + self.get_distance(self.tour[j], self.tour[(j + 1) % self.num_cities]) > self.get_distance(self.tour[i], self.tour[j]) + self.get_distance(self.tour[i + 1], self.tour[(j + 1) % self.num_cities]):
                        self.tour[i + 1:j + 1] = reversed(self.tour[i + 1:j + 1])
                        improved = True

    def solve(self):
        self.improve_tour()
        return self.tour


# In[11]:


import matplotlib.pyplot as plt
import time
import random
import itertools
import math

# Generate worst-case input: all cities on a straight line
def generate_worst_case_input(size):
    cities = [(i, 0) for i in range(size)]
    return cities

# Generate best-case input: cities arranged in a circle
def generate_best_case_input(size):
    angle = 2 * math.pi / size
    cities = [(math.cos(i * angle), math.sin(i * angle)) for i in range(size)]
    return cities

# Generate average-case input: randomly generated cities
def generate_average_case_input(size):
    cities = [(random.random(), random.random()) for _ in range(size)]
    return cities

# Measure execution time for a given input size
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanTwoOpt(cities)
    start_time = time.time()
    tour = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return tour, execution_time

# Generate input sizes (number of cities)
input_sizes = [5, 10, 15, 20,25]

# Generate worst-case execution times and tours
worst_case_times = []
worst_case_tours = []
for size in input_sizes:
    cities = generate_worst_case_input(size)
    tour, execution_time = measure_execution_time(cities)
    worst_case_times.append(execution_time)
    worst_case_tours.append(tour)

# Generate best-case execution times and tours
best_case_times = []
best_case_tours = []
for size in input_sizes:
    cities = generate_best_case_input(size)
    tour, execution_time = measure_execution_time(cities)
    best_case_times.append(execution_time)
    best_case_tours.append(tour)

# Generate average-case execution times and tours
average_case_times = []
average_case_tours = []
for size in input_sizes:
    cities = generate_average_case_input(size)
    tour, execution_time = measure_execution_time(cities)
    average_case_times.append(execution_time)
    average_case_tours.append(tour)

# Plot the execution times for worst-case, best-case, and average-case scenarios
plt.plot(input_sizes, worst_case_times, label="Worst Case")
plt.plot(input_sizes, best_case_times, label="Best Case")
plt.plot(input_sizes, average_case_times, label="Average Case")
plt.xlabel("Number of Cities")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Times for Different Cases")
plt.legend()
plt.show()


# In[12]:


import matplotlib.pyplot as plt
import time
import random
import itertools
import math

# Generate worst-case input: all cities on a straight line
def generate_worst_case_input(size):
    cities = [(i, 0) for i in range(size)]
    return cities

# Generate best-case input: cities arranged in a circle
def generate_best_case_input(size):
    angle = 2 * math.pi / size
    cities = [(math.cos(i * angle), math.sin(i * angle)) for i in range(size)]
    return cities

# Generate average-case input: randomly generated cities
def generate_average_case_input(size):
    cities = [(random.random(), random.random()) for _ in range(size)]
    return cities

# Measure execution time for a given input size
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanTwoOpt(cities)
    start_time = time.time()
    tour = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return tour, execution_time

# Generate input sizes (number of cities)
input_sizes = [10,20,30,40,50]

# Generate worst-case execution times and tours
worst_case_times = []
worst_case_tours = []
for size in input_sizes:
    cities = generate_worst_case_input(size)
    tour, execution_time = measure_execution_time(cities)
    worst_case_times.append(execution_time)
    worst_case_tours.append(tour)

# Generate best-case execution times and tours
best_case_times = []
best_case_tours = []
for size in input_sizes:
    cities = generate_best_case_input(size)
    tour, execution_time = measure_execution_time(cities)
    best_case_times.append(execution_time)
    best_case_tours.append(tour)

# Generate average-case execution times and tours
average_case_times = []
average_case_tours = []
for size in input_sizes:
    cities = generate_average_case_input(size)
    tour, execution_time = measure_execution_time(cities)
    average_case_times.append(execution_time)
    average_case_tours.append(tour)

# Plot the execution times for worst-case, best-case, and average-case scenarios
plt.plot(input_sizes, worst_case_times, label="Worst Case")
plt.plot(input_sizes, best_case_times, label="Best Case")
plt.plot(input_sizes, average_case_times, label="Average Case")
plt.xlabel("Number of Cities")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Times for Different Cases")
plt.legend()
plt.show()


# In[11]:


import matplotlib.pyplot as plt
import time
import random
import itertools
import math

# Generate worst-case input: all cities on a straight line
def generate_worst_case_input(size):
    cities = [(i, 0) for i in range(size)]
    return cities

# Generate best-case input: cities arranged in a circle
def generate_best_case_input(size):
    angle = 2 * math.pi / size
    cities = [(math.cos(i * angle), math.sin(i * angle)) for i in range(size)]
    return cities

# Generate average-case input: randomly generated cities
def generate_average_case_input(size):
    cities = [(random.random(), random.random()) for _ in range(size)]
    return cities

# Measure execution time for a given input size
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanTwoOpt(cities)
    start_time = time.time()
    tour = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return tour, execution_time

# Generate input sizes (number of cities)
input_sizes = [30,60,90,120,150]

# Generate worst-case execution times and tours
worst_case_times = []
worst_case_tours = []
for size in input_sizes:
    cities = generate_worst_case_input(size)
    tour, execution_time = measure_execution_time(cities)
    worst_case_times.append(execution_time)
    worst_case_tours.append(tour)

# Generate best-case execution times and tours
best_case_times = []
best_case_tours = []
for size in input_sizes:
    cities = generate_best_case_input(size)
    tour, execution_time = measure_execution_time(cities)
    best_case_times.append(execution_time)
    best_case_tours.append(tour)

# Generate average-case execution times and tours
average_case_times = []
average_case_tours = []
for size in input_sizes:
    cities = generate_average_case_input(size)
    tour, execution_time = measure_execution_time(cities)
    average_case_times.append(execution_time)
    average_case_tours.append(tour)

# Plot the execution times for worst-case, best-case, and average-case scenarios
plt.plot(input_sizes, worst_case_times, label="Worst Case")
plt.plot(input_sizes, best_case_times, label="Best Case")
plt.plot(input_sizes, average_case_times, label="Average Case")
plt.xlabel("Number of Cities")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Times for Different Cases")
plt.legend()
plt.show()


# In[12]:


import matplotlib.pyplot as plt
import time
import random
import itertools
import math

# Generate worst-case input: all cities on a straight line
def generate_worst_case_input(size):
    cities = [(i, 0) for i in range(size)]
    return cities

# Generate best-case input: cities arranged in a circle
def generate_best_case_input(size):
    angle = 2 * math.pi / size
    cities = [(math.cos(i * angle), math.sin(i * angle)) for i in range(size)]
    return cities

# Generate average-case input: randomly generated cities
def generate_average_case_input(size):
    cities = [(random.random(), random.random()) for _ in range(size)]
    return cities

# Measure execution time for a given input size
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanTwoOpt(cities)
    start_time = time.time()
    tour = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return tour, execution_time

# Generate input sizes (number of cities)
input_sizes = [50,100,150,200,250]

# Generate worst-case execution times and tours
worst_case_times = []
worst_case_tours = []
for size in input_sizes:
    cities = generate_worst_case_input(size)
    tour, execution_time = measure_execution_time(cities)
    worst_case_times.append(execution_time)
    worst_case_tours.append(tour)

# Generate best-case execution times and tours
best_case_times = []
best_case_tours = []
for size in input_sizes:
    cities = generate_best_case_input(size)
    tour, execution_time = measure_execution_time(cities)
    best_case_times.append(execution_time)
    best_case_tours.append(tour)

# Generate average-case execution times and tours
average_case_times = []
average_case_tours = []
for size in input_sizes:
    cities = generate_average_case_input(size)
    tour, execution_time = measure_execution_time(cities)
    average_case_times.append(execution_time)
    average_case_tours.append(tour)

# Plot the execution times for worst-case, best-case, and average-case scenarios
plt.plot(input_sizes, worst_case_times, label="Worst Case")
plt.plot(input_sizes, best_case_times, label="Best Case")
plt.plot(input_sizes, average_case_times, label="Average Case")
plt.xlabel("Number of Cities")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Times for Different Cases")
plt.legend()
plt.show()


# In[13]:


import matplotlib.pyplot as plt
import time
import random
import itertools
import math

# Generate worst-case input: all cities on a straight line
def generate_worst_case_input(size):
    cities = [(i, 0) for i in range(size)]
    return cities

# Generate best-case input: cities arranged in a circle
def generate_best_case_input(size):
    angle = 2 * math.pi / size
    cities = [(math.cos(i * angle), math.sin(i * angle)) for i in range(size)]
    return cities

# Generate average-case input: randomly generated cities
def generate_average_case_input(size):
    cities = [(random.random(), random.random()) for _ in range(size)]
    return cities

# Measure execution time for a given input size
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanTwoOpt(cities)
    start_time = time.time()
    tour = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return tour, execution_time

# Generate input sizes (number of cities)
input_sizes = [100,200,300,400,500]

# Generate worst-case execution times and tours
worst_case_times = []
worst_case_tours = []
for size in input_sizes:
    cities = generate_worst_case_input(size)
    tour, execution_time = measure_execution_time(cities)
    worst_case_times.append(execution_time)
    worst_case_tours.append(tour)

# Generate best-case execution times and tours
best_case_times = []
best_case_tours = []
for size in input_sizes:
    cities = generate_best_case_input(size)
    tour, execution_time = measure_execution_time(cities)
    best_case_times.append(execution_time)
    best_case_tours.append(tour)

# Generate average-case execution times and tours
average_case_times = []
average_case_tours = []
for size in input_sizes:
    cities = generate_average_case_input(size)
    tour, execution_time = measure_execution_time(cities)
    average_case_times.append(execution_time)
    average_case_tours.append(tour)

# Plot the execution times for worst-case, best-case, and average-case scenarios
plt.plot(input_sizes, worst_case_times, label="Worst Case")
plt.plot(input_sizes, best_case_times, label="Best Case")
plt.plot(input_sizes, average_case_times, label="Average Case")
plt.xlabel("Number of Cities")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Times for Different Cases")
plt.legend()
plt.show()


# In[14]:


import matplotlib.pyplot as plt
import time
import random
import itertools
import math

# Generate worst-case input: all cities on a straight line
def generate_worst_case_input(size):
    cities = [(i, 0) for i in range(size)]
    return cities

# Generate best-case input: cities arranged in a circle
def generate_best_case_input(size):
    angle = 2 * math.pi / size
    cities = [(math.cos(i * angle), math.sin(i * angle)) for i in range(size)]
    return cities

# Generate average-case input: randomly generated cities
def generate_average_case_input(size):
    cities = [(random.random(), random.random()) for _ in range(size)]
    return cities

# Measure execution time for a given input size
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanTwoOpt(cities)
    start_time = time.time()
    tour = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return tour, execution_time

# Generate input sizes (number of cities)
input_sizes = [30,40,50,60,70,80]

# Generate worst-case execution times and tours
worst_case_times = []
worst_case_tours = []
for size in input_sizes:
    cities = generate_worst_case_input(size)
    tour, execution_time = measure_execution_time(cities)
    worst_case_times.append(execution_time)
    worst_case_tours.append(tour)

# Generate best-case execution times and tours
best_case_times = []
best_case_tours = []
for size in input_sizes:
    cities = generate_best_case_input(size)
    tour, execution_time = measure_execution_time(cities)
    best_case_times.append(execution_time)
    best_case_tours.append(tour)

# Generate average-case execution times and tours
average_case_times = []
average_case_tours = []
for size in input_sizes:
    cities = generate_average_case_input(size)
    tour, execution_time = measure_execution_time(cities)
    average_case_times.append(execution_time)
    average_case_tours.append(tour)

# Plot the execution times for worst-case, best-case, and average-case scenarios
plt.plot(input_sizes, worst_case_times, label="Worst Case")
plt.plot(input_sizes, best_case_times, label="Best Case")
plt.plot(input_sizes, average_case_times, label="Average Case")
plt.xlabel("Number of Cities")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Times for Different Cases")
plt.legend()
plt.show()


# In[15]:


import matplotlib.pyplot as plt
import time
import random
import itertools
import math

# Generate worst-case input: all cities on a straight line
def generate_worst_case_input(size):
    cities = [(i, 0) for i in range(size)]
    return cities

# Generate best-case input: cities arranged in a circle
def generate_best_case_input(size):
    angle = 2 * math.pi / size
    cities = [(math.cos(i * angle), math.sin(i * angle)) for i in range(size)]
    return cities

# Generate average-case input: randomly generated cities
def generate_average_case_input(size):
    cities = [(random.random(), random.random()) for _ in range(size)]
    return cities

# Measure execution time for a given input size
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanTwoOpt(cities)
    start_time = time.time()
    tour = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return tour, execution_time

# Generate input sizes (number of cities)
input_sizes = [300,600,900,1200,1500]

# Generate worst-case execution times and tours
worst_case_times = []
worst_case_tours = []
for size in input_sizes:
    cities = generate_worst_case_input(size)
    tour, execution_time = measure_execution_time(cities)
    worst_case_times.append(execution_time)
    worst_case_tours.append(tour)

# Generate best-case execution times and tours
best_case_times = []
best_case_tours = []
for size in input_sizes:
    cities = generate_best_case_input(size)
    tour, execution_time = measure_execution_time(cities)
    best_case_times.append(execution_time)
    best_case_tours.append(tour)

# Generate average-case execution times and tours
average_case_times = []
average_case_tours = []
for size in input_sizes:
    cities = generate_average_case_input(size)
    tour, execution_time = measure_execution_time(cities)
    average_case_times.append(execution_time)
    average_case_tours.append(tour)

# Plot the execution times for worst-case, best-case, and average-case scenarios
plt.plot(input_sizes, worst_case_times, label="Worst Case")
plt.plot(input_sizes, best_case_times, label="Best Case")
plt.plot(input_sizes, average_case_times, label="Average Case")
plt.xlabel("Number of Cities")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Times for Different Cases")
plt.legend()
plt.show()


# In[ ]:




