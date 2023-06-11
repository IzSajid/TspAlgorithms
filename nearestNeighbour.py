#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import random

class TravelingSalesmanNearestNeighbor:
    def __init__(self, cities):
        self.cities = cities
        self.num_cities = len(cities)

    def get_distance(self, city1, city2):
        return np.linalg.norm(np.array(city1) - np.array(city2))

    def solve(self):
        unvisited_cities = set(range(self.num_cities))
        current_city = random.sample(range(self.num_cities), 1)[0]
        path = [current_city]
        unvisited_cities.remove(current_city)
        total_distance = 0

        while unvisited_cities:
            nearest_city = min(unvisited_cities, key=lambda city: self.get_distance(self.cities[current_city], self.cities[city]))
            path.append(nearest_city)
            unvisited_cities.remove(nearest_city)
            total_distance += self.get_distance(self.cities[current_city], self.cities[nearest_city])
            current_city = nearest_city

        total_distance += self.get_distance(self.cities[path[-1]], self.cities[path[0]])

        return total_distance, [self.cities[i] for i in path]


# In[6]:


import time

# Function to measure the execution time of your algorithm
def measure_time_complexity():
    start_time = time.time()

    # Your algorithm code here
    cities = [(10, 10), (20, 30), (55, 25), (70, 48), (40, 60)]
    tsp_nn = TravelingSalesmanNearestNeighbor(cities)
    min_distance, min_path = tsp_nn.solve()

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time)
    print("Minimum distance:", min_distance)
    print("Minimum path:", min_path)

# Call the function to measure the time complexity
measure_time_complexity()


# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import time

# Generate worst-case input: Cities evenly distributed in a grid-like pattern
def generate_worst_case_input(size):
    cities_per_row = int(np.sqrt(size))
    cities = []
    for i in range(cities_per_row):
        for j in range(cities_per_row):
            x = i * (1.0 / cities_per_row)
            y = j * (1.0 / cities_per_row)
            cities.append((x, y))
    return cities

# Generate best-case input: Cities randomly distributed in a small area
def generate_best_case_input(size):
    cities = np.random.rand(size, 2) * 0.1  # Randomly generate cities in the range (0, 0) to (0.1, 0.1)
    return cities.tolist()

# Generate average-case input: Randomly generated cities
def generate_average_case_input(size):
    cities = np.random.rand(size, 2)
    return cities.tolist()

# Measure execution time for a given input size using existing algorithm
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanNearestNeighbor(cities)
    start_time = time.time()
    min_distance, min_path = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, min_path, execution_time

# Generate input sizes (number of cities)
input_sizes = [10, 20, 30, 40, 50]

# Generate worst-case execution times
worst_case_times = []
for size in input_sizes:
    cities = generate_worst_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    worst_case_times.append(execution_time)

# Generate best-case execution times
best_case_times = []
for size in input_sizes:
    cities = generate_best_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    best_case_times.append(execution_time)

# Generate average-case execution times
average_case_times = []
for size in input_sizes:
    cities = generate_average_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    average_case_times.append(execution_time)

# Plot the graphs
plt.plot(input_sizes, worst_case_times, label="Worst Case")
plt.plot(input_sizes, best_case_times, label="Best Case")
plt.plot(input_sizes, average_case_times, label="Average Case")
plt.xlabel("Number of Cities")
plt.ylabel("Execution Time")
plt.legend()
plt.show()


# In[13]:


import numpy as np
import matplotlib.pyplot as plt
import time

# Generate worst-case input: Cities evenly distributed in a grid-like pattern
def generate_worst_case_input(size):
    cities_per_row = int(np.sqrt(size))
    cities = []
    for i in range(cities_per_row):
        for j in range(cities_per_row):
            x = i * (1.0 / cities_per_row)
            y = j * (1.0 / cities_per_row)
            cities.append((x, y))
    return cities

# Generate best-case input: Cities randomly distributed in a small area
def generate_best_case_input(size):
    cities = np.random.rand(size, 2) * 0.1  # Randomly generate cities in the range (0, 0) to (0.1, 0.1)
    return cities.tolist()

# Generate average-case input: Randomly generated cities
def generate_average_case_input(size):
    cities = np.random.rand(size, 2)
    return cities.tolist()

# Measure execution time for a given input size using existing algorithm
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanNearestNeighbor(cities)
    start_time = time.time()
    min_distance, min_path = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, min_path, execution_time

# Generate input sizes (number of cities)
input_sizes = [10, 20, 30, 40, 50,60]

# Generate worst-case execution times
worst_case_times = []
for size in input_sizes:
    cities = generate_worst_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    worst_case_times.append(execution_time)

# Generate best-case execution times
best_case_times = []
for size in input_sizes:
    cities = generate_best_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    best_case_times.append(execution_time)

# Generate average-case execution times
average_case_times = []
for size in input_sizes:
    cities = generate_average_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    average_case_times.append(execution_time)

# Plot the graphs
plt.plot(input_sizes, worst_case_times, label="Worst Case")
plt.plot(input_sizes, best_case_times, label="Best Case")
plt.plot(input_sizes, average_case_times, label="Average Case")
plt.xlabel("Number of Cities")
plt.ylabel("Execution Time")
plt.legend()
plt.show()


# In[19]:


import numpy as np
import matplotlib.pyplot as plt
import time

# Generate worst-case input: Cities evenly distributed in a grid-like pattern
def generate_worst_case_input(size):
    cities_per_row = int(np.sqrt(size))
    cities = []
    for i in range(cities_per_row):
        for j in range(cities_per_row):
            x = i * (1.0 / cities_per_row)
            y = j * (1.0 / cities_per_row)
            cities.append((x, y))
    return cities

# Generate best-case input: Cities randomly distributed in a small area
def generate_best_case_input(size):
    cities = np.random.rand(size, 2) * 0.1  # Randomly generate cities in the range (0, 0) to (0.1, 0.1)
    return cities.tolist()

# Generate average-case input: Randomly generated cities
def generate_average_case_input(size):
    cities = np.random.rand(size, 2)
    return cities.tolist()

# Measure execution time for a given input size using existing algorithm
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanNearestNeighbor(cities)
    start_time = time.time()
    min_distance, min_path = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, min_path, execution_time

# Generate input sizes (number of cities)
input_sizes = [10,20,30, 40, 50,60,70]

# Generate worst-case execution times
worst_case_times = []
for size in input_sizes:
    cities = generate_worst_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    worst_case_times.append(execution_time)

# Generate best-case execution times
best_case_times = []
for size in input_sizes:
    cities = generate_best_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    best_case_times.append(execution_time)

# Generate average-case execution times
average_case_times = []
for size in input_sizes:
    cities = generate_average_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    average_case_times.append(execution_time)

# Plot the graphs
plt.plot(input_sizes, worst_case_times, label="Worst Case")
plt.plot(input_sizes, best_case_times, label="Best Case")
plt.plot(input_sizes, average_case_times, label="Average Case")
plt.xlabel("Number of Cities")
plt.ylabel("Execution Time")
plt.legend()
plt.show()


# In[31]:


import numpy as np
import matplotlib.pyplot as plt
import time

# Generate worst-case input: Cities evenly distributed in a grid-like pattern
def generate_worst_case_input(size):
    cities_per_row = int(np.sqrt(size))
    cities = []
    for i in range(cities_per_row):
        for j in range(cities_per_row):
            x = i * (1.0 / cities_per_row)
            y = j * (1.0 / cities_per_row)
            cities.append((x, y))
    return cities

# Generate best-case input: Cities randomly distributed in a small area
def generate_best_case_input(size):
    cities = np.random.rand(size, 2) * 0.1  # Randomly generate cities in the range (0, 0) to (0.1, 0.1)
    return cities.tolist()

# Generate average-case input: Randomly generated cities
def generate_average_case_input(size):
    cities = np.random.rand(size, 2)
    return cities.tolist()

# Measure execution time for a given input size using existing algorithm
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanNearestNeighbor(cities)
    start_time = time.time()
    min_distance, min_path = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, min_path, execution_time

# Generate input sizes (number of cities)
input_sizes = [10,20,30, 40, 50,60,70,80]

# Generate worst-case execution times
worst_case_times = []
for size in input_sizes:
    cities = generate_worst_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    worst_case_times.append(execution_time)

# Generate best-case execution times
best_case_times = []
for size in input_sizes:
    cities = generate_best_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    best_case_times.append(execution_time)

# Generate average-case execution times
average_case_times = []
for size in input_sizes:
    cities = generate_average_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    average_case_times.append(execution_time)

# Plot the graphs
plt.plot(input_sizes, worst_case_times, label="Worst Case")
plt.plot(input_sizes, best_case_times, label="Best Case")
plt.plot(input_sizes, average_case_times, label="Average Case")
plt.xlabel("Number of Cities")
plt.ylabel("Execution Time")
plt.legend()
plt.show()


# In[21]:


import numpy as np
import matplotlib.pyplot as plt
import time

# Generate worst-case input: Cities evenly distributed in a grid-like pattern
def generate_worst_case_input(size):
    cities_per_row = int(np.sqrt(size))
    cities = []
    for i in range(cities_per_row):
        for j in range(cities_per_row):
            x = i * (1.0 / cities_per_row)
            y = j * (1.0 / cities_per_row)
            cities.append((x, y))
    return cities

# Generate best-case input: Cities randomly distributed in a small area
def generate_best_case_input(size):
    cities = np.random.rand(size, 2) * 0.1  # Randomly generate cities in the range (0, 0) to (0.1, 0.1)
    return cities.tolist()

# Generate average-case input: Randomly generated cities
def generate_average_case_input(size):
    cities = np.random.rand(size, 2)
    return cities.tolist()

# Measure execution time for a given input size using existing algorithm
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanNearestNeighbor(cities)
    start_time = time.time()
    min_distance, min_path = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, min_path, execution_time

# Generate input sizes (number of cities)
input_sizes = [10,20,30, 40, 50,60,70,90]

# Generate worst-case execution times
worst_case_times = []
for size in input_sizes:
    cities = generate_worst_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    worst_case_times.append(execution_time)

# Generate best-case execution times
best_case_times = []
for size in input_sizes:
    cities = generate_best_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    best_case_times.append(execution_time)

# Generate average-case execution times
average_case_times = []
for size in input_sizes:
    cities = generate_average_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    average_case_times.append(execution_time)

# Plot the graphs
plt.plot(input_sizes, worst_case_times, label="Worst Case")
plt.plot(input_sizes, best_case_times, label="Best Case")
plt.plot(input_sizes, average_case_times, label="Average Case")
plt.xlabel("Number of Cities")
plt.ylabel("Execution Time")
plt.legend()
plt.show()


# In[22]:


import numpy as np
import matplotlib.pyplot as plt
import time

# Generate worst-case input: Cities evenly distributed in a grid-like pattern
def generate_worst_case_input(size):
    cities_per_row = int(np.sqrt(size))
    cities = []
    for i in range(cities_per_row):
        for j in range(cities_per_row):
            x = i * (1.0 / cities_per_row)
            y = j * (1.0 / cities_per_row)
            cities.append((x, y))
    return cities

# Generate best-case input: Cities randomly distributed in a small area
def generate_best_case_input(size):
    cities = np.random.rand(size, 2) * 0.1  # Randomly generate cities in the range (0, 0) to (0.1, 0.1)
    return cities.tolist()

# Generate average-case input: Randomly generated cities
def generate_average_case_input(size):
    cities = np.random.rand(size, 2)
    return cities.tolist()

# Measure execution time for a given input size using existing algorithm
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanNearestNeighbor(cities)
    start_time = time.time()
    min_distance, min_path = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, min_path, execution_time

# Generate input sizes (number of cities)
input_sizes = [10,20,30, 40, 50,60,70,80,90,100]

# Generate worst-case execution times
worst_case_times = []
for size in input_sizes:
    cities = generate_worst_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    worst_case_times.append(execution_time)

# Generate best-case execution times
best_case_times = []
for size in input_sizes:
    cities = generate_best_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    best_case_times.append(execution_time)

# Generate average-case execution times
average_case_times = []
for size in input_sizes:
    cities = generate_average_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    average_case_times.append(execution_time)

# Plot the graphs
plt.plot(input_sizes, worst_case_times, label="Worst Case")
plt.plot(input_sizes, best_case_times, label="Best Case")
plt.plot(input_sizes, average_case_times, label="Average Case")
plt.xlabel("Number of Cities")
plt.ylabel("Execution Time")
plt.legend()
plt.show()


# In[23]:


import numpy as np
import matplotlib.pyplot as plt
import time

# Generate worst-case input: Cities evenly distributed in a grid-like pattern
def generate_worst_case_input(size):
    cities_per_row = int(np.sqrt(size))
    cities = []
    for i in range(cities_per_row):
        for j in range(cities_per_row):
            x = i * (1.0 / cities_per_row)
            y = j * (1.0 / cities_per_row)
            cities.append((x, y))
    return cities

# Generate best-case input: Cities randomly distributed in a small area
def generate_best_case_input(size):
    cities = np.random.rand(size, 2) * 0.1  # Randomly generate cities in the range (0, 0) to (0.1, 0.1)
    return cities.tolist()

# Generate average-case input: Randomly generated cities
def generate_average_case_input(size):
    cities = np.random.rand(size, 2)
    return cities.tolist()

# Measure execution time for a given input size using existing algorithm
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanNearestNeighbor(cities)
    start_time = time.time()
    min_distance, min_path = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, min_path, execution_time

# Generate input sizes (number of cities)
input_sizes = [40,60,80,100,120,140,160]

# Generate worst-case execution times
worst_case_times = []
for size in input_sizes:
    cities = generate_worst_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    worst_case_times.append(execution_time)

# Generate best-case execution times
best_case_times = []
for size in input_sizes:
    cities = generate_best_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    best_case_times.append(execution_time)

# Generate average-case execution times
average_case_times = []
for size in input_sizes:
    cities = generate_average_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    average_case_times.append(execution_time)

# Plot the graphs
plt.plot(input_sizes, worst_case_times, label="Worst Case")
plt.plot(input_sizes, best_case_times, label="Best Case")
plt.plot(input_sizes, average_case_times, label="Average Case")
plt.xlabel("Number of Cities")
plt.ylabel("Execution Time")
plt.legend()
plt.show()


# In[24]:


import numpy as np
import matplotlib.pyplot as plt
import time

# Generate worst-case input: Cities evenly distributed in a grid-like pattern
def generate_worst_case_input(size):
    cities_per_row = int(np.sqrt(size))
    cities = []
    for i in range(cities_per_row):
        for j in range(cities_per_row):
            x = i * (1.0 / cities_per_row)
            y = j * (1.0 / cities_per_row)
            cities.append((x, y))
    return cities

# Generate best-case input: Cities randomly distributed in a small area
def generate_best_case_input(size):
    cities = np.random.rand(size, 2) * 0.1  # Randomly generate cities in the range (0, 0) to (0.1, 0.1)
    return cities.tolist()

# Generate average-case input: Randomly generated cities
def generate_average_case_input(size):
    cities = np.random.rand(size, 2)
    return cities.tolist()

# Measure execution time for a given input size using existing algorithm
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanNearestNeighbor(cities)
    start_time = time.time()
    min_distance, min_path = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, min_path, execution_time

# Generate input sizes (number of cities)
input_sizes = [100,130,160,190,220,250]

# Generate worst-case execution times
worst_case_times = []
for size in input_sizes:
    cities = generate_worst_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    worst_case_times.append(execution_time)

# Generate best-case execution times
best_case_times = []
for size in input_sizes:
    cities = generate_best_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    best_case_times.append(execution_time)

# Generate average-case execution times
average_case_times = []
for size in input_sizes:
    cities = generate_average_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    average_case_times.append(execution_time)

# Plot the graphs
plt.plot(input_sizes, worst_case_times, label="Worst Case")
plt.plot(input_sizes, best_case_times, label="Best Case")
plt.plot(input_sizes, average_case_times, label="Average Case")
plt.xlabel("Number of Cities")
plt.ylabel("Execution Time")
plt.legend()
plt.show()


# In[26]:


import numpy as np
import matplotlib.pyplot as plt
import time

# Generate worst-case input: Cities evenly distributed in a grid-like pattern
def generate_worst_case_input(size):
    cities_per_row = int(np.sqrt(size))
    cities = []
    for i in range(cities_per_row):
        for j in range(cities_per_row):
            x = i * (1.0 / cities_per_row)
            y = j * (1.0 / cities_per_row)
            cities.append((x, y))
    return cities

# Generate best-case input: Cities randomly distributed in a small area
def generate_best_case_input(size):
    cities = np.random.rand(size, 2) * 0.1  # Randomly generate cities in the range (0, 0) to (0.1, 0.1)
    return cities.tolist()

# Generate average-case input: Randomly generated cities
def generate_average_case_input(size):
    cities = np.random.rand(size, 2)
    return cities.tolist()

# Measure execution time for a given input size using existing algorithm
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanNearestNeighbor(cities)
    start_time = time.time()
    min_distance, min_path = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, min_path, execution_time

# Generate input sizes (number of cities)
input_sizes = [100,150,200,250,300]

# Generate worst-case execution times
worst_case_times = []
for size in input_sizes:
    cities = generate_worst_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    worst_case_times.append(execution_time)

# Generate best-case execution times
best_case_times = []
for size in input_sizes:
    cities = generate_best_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    best_case_times.append(execution_time)

# Generate average-case execution times
average_case_times = []
for size in input_sizes:
    cities = generate_average_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    average_case_times.append(execution_time)

# Plot the graphs
plt.plot(input_sizes, worst_case_times, label="Worst Case")
plt.plot(input_sizes, best_case_times, label="Best Case")
plt.plot(input_sizes, average_case_times, label="Average Case")
plt.xlabel("Number of Cities")
plt.ylabel("Execution Time")
plt.legend()
plt.show()


# In[28]:


import numpy as np
import matplotlib.pyplot as plt
import time

# Generate worst-case input: Cities evenly distributed in a grid-like pattern
def generate_worst_case_input(size):
    cities_per_row = int(np.sqrt(size))
    cities = []
    for i in range(cities_per_row):
        for j in range(cities_per_row):
            x = i * (1.0 / cities_per_row)
            y = j * (1.0 / cities_per_row)
            cities.append((x, y))
    return cities

# Generate best-case input: Cities randomly distributed in a small area
def generate_best_case_input(size):
    cities = np.random.rand(size, 2) * 0.1  # Randomly generate cities in the range (0, 0) to (0.1, 0.1)
    return cities.tolist()

# Generate average-case input: Randomly generated cities
def generate_average_case_input(size):
    cities = np.random.rand(size, 2)
    return cities.tolist()

# Measure execution time for a given input size using existing algorithm
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanNearestNeighbor(cities)
    start_time = time.time()
    min_distance, min_path = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, min_path, execution_time

# Generate input sizes (number of cities)
input_sizes = [100,200,300,400,500]

# Generate worst-case execution times
worst_case_times = []
for size in input_sizes:
    cities = generate_worst_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    worst_case_times.append(execution_time)

# Generate best-case execution times
best_case_times = []
for size in input_sizes:
    cities = generate_best_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    best_case_times.append(execution_time)

# Generate average-case execution times
average_case_times = []
for size in input_sizes:
    cities = generate_average_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    average_case_times.append(execution_time)

# Plot the graphs
plt.plot(input_sizes, worst_case_times, label="Worst Case")
plt.plot(input_sizes, best_case_times, label="Best Case")
plt.plot(input_sizes, average_case_times, label="Average Case")
plt.xlabel("Number of Cities")
plt.ylabel("Execution Time")
plt.legend()
plt.show()


# In[29]:


import numpy as np
import matplotlib.pyplot as plt
import time

# Generate worst-case input: Cities evenly distributed in a grid-like pattern
def generate_worst_case_input(size):
    cities_per_row = int(np.sqrt(size))
    cities = []
    for i in range(cities_per_row):
        for j in range(cities_per_row):
            x = i * (1.0 / cities_per_row)
            y = j * (1.0 / cities_per_row)
            cities.append((x, y))
    return cities

# Generate best-case input: Cities randomly distributed in a small area
def generate_best_case_input(size):
    cities = np.random.rand(size, 2) * 0.1  # Randomly generate cities in the range (0, 0) to (0.1, 0.1)
    return cities.tolist()

# Generate average-case input: Randomly generated cities
def generate_average_case_input(size):
    cities = np.random.rand(size, 2)
    return cities.tolist()

# Measure execution time for a given input size using existing algorithm
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanNearestNeighbor(cities)
    start_time = time.time()
    min_distance, min_path = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, min_path, execution_time

# Generate input sizes (number of cities)
input_sizes = [200,400,600,800,1000]

# Generate worst-case execution times
worst_case_times = []
for size in input_sizes:
    cities = generate_worst_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    worst_case_times.append(execution_time)

# Generate best-case execution times
best_case_times = []
for size in input_sizes:
    cities = generate_best_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    best_case_times.append(execution_time)

# Generate average-case execution times
average_case_times = []
for size in input_sizes:
    cities = generate_average_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    average_case_times.append(execution_time)

# Plot the graphs
plt.plot(input_sizes, worst_case_times, label="Worst Case")
plt.plot(input_sizes, best_case_times, label="Best Case")
plt.plot(input_sizes, average_case_times, label="Average Case")
plt.xlabel("Number of Cities")
plt.ylabel("Execution Time")
plt.legend()
plt.show()


# In[38]:


import numpy as np
import matplotlib.pyplot as plt
import time

# Generate worst-case input: Cities evenly distributed in a grid-like pattern
def generate_worst_case_input(size):
    cities_per_row = int(np.sqrt(size))
    cities = []
    for i in range(cities_per_row):
        for j in range(cities_per_row):
            x = i * (1.0 / cities_per_row)
            y = j * (1.0 / cities_per_row)
            cities.append((x, y))
    return cities

# Generate best-case input: Cities randomly distributed in a small area
def generate_best_case_input(size):
    cities = np.random.rand(size, 2) * 0.1  # Randomly generate cities in the range (0, 0) to (0.1, 0.1)
    return cities.tolist()

# Generate average-case input: Randomly generated cities
def generate_average_case_input(size):
    cities = np.random.rand(size, 2)
    return cities.tolist()

# Measure execution time for a given input size using existing algorithm
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanNearestNeighbor(cities)
    start_time = time.time()
    min_distance, min_path = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, min_path, execution_time

# Generate input sizes (number of cities)
input_sizes = [100,150,200,250,300]

# Generate worst-case execution times
worst_case_times = []
for size in input_sizes:
    cities = generate_worst_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    worst_case_times.append(execution_time)

# Generate best-case execution times
best_case_times = []
for size in input_sizes:
    cities = generate_best_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    best_case_times.append(execution_time)

# Generate average-case execution times
average_case_times = []
for size in input_sizes:
    cities = generate_average_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    average_case_times.append(execution_time)

# Plot the graphs
plt.plot(input_sizes, worst_case_times, label="Worst Case")
plt.plot(input_sizes, best_case_times, label="Best Case")
plt.plot(input_sizes, average_case_times, label="Average Case")
plt.xlabel("Number of Cities")
plt.ylabel("Execution Time")
plt.legend()
plt.show()


# In[39]:


import numpy as np
import matplotlib.pyplot as plt
import time

# Generate worst-case input: Cities evenly distributed in a grid-like pattern
def generate_worst_case_input(size):
    cities_per_row = int(np.sqrt(size))
    cities = []
    for i in range(cities_per_row):
        for j in range(cities_per_row):
            x = i * (1.0 / cities_per_row)
            y = j * (1.0 / cities_per_row)
            cities.append((x, y))
    return cities

# Generate best-case input: Cities randomly distributed in a small area
def generate_best_case_input(size):
    cities = np.random.rand(size, 2) * 0.1  # Randomly generate cities in the range (0, 0) to (0.1, 0.1)
    return cities.tolist()

# Generate average-case input: Randomly generated cities
def generate_average_case_input(size):
    cities = np.random.rand(size, 2)
    return cities.tolist()

# Measure execution time for a given input size using existing algorithm
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanNearestNeighbor(cities)
    start_time = time.time()
    min_distance, min_path = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, min_path, execution_time

# Generate input sizes (number of cities)
input_sizes = [400,800,1200,1600,2000]

# Generate worst-case execution times
worst_case_times = []
for size in input_sizes:
    cities = generate_worst_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    worst_case_times.append(execution_time)

# Generate best-case execution times
best_case_times = []
for size in input_sizes:
    cities = generate_best_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    best_case_times.append(execution_time)

# Generate average-case execution times
average_case_times = []
for size in input_sizes:
    cities = generate_average_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    average_case_times.append(execution_time)

# Plot the graphs
plt.plot(input_sizes, worst_case_times, label="Worst Case")
plt.plot(input_sizes, best_case_times, label="Best Case")
plt.plot(input_sizes, average_case_times, label="Average Case")
plt.xlabel("Number of Cities")
plt.ylabel("Execution Time")
plt.legend()
plt.show()


# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import time

# Generate worst-case input: Cities evenly distributed in a grid-like pattern
def generate_worst_case_input(size):
    cities_per_row = int(np.sqrt(size))
    cities = []
    for i in range(cities_per_row):
        for j in range(cities_per_row):
            x = i * (1.0 / cities_per_row)
            y = j * (1.0 / cities_per_row)
            cities.append((x, y))
    return cities

# Generate best-case input: Cities randomly distributed in a small area
def generate_best_case_input(size):
    cities = np.random.rand(size, 2) * 0.1  # Randomly generate cities in the range (0, 0) to (0.1, 0.1)
    return cities.tolist()

# Generate average-case input: Randomly generated cities
def generate_average_case_input(size):
    cities = np.random.rand(size, 2)
    return cities.tolist()

# Measure execution time for a given input size using existing algorithm
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanNearestNeighbor(cities)
    start_time = time.time()
    min_distance, min_path = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, min_path, execution_time

# Generate input sizes (number of cities)
input_sizes = [300,600,900,1200,1500]

# Generate worst-case execution times
worst_case_times = []
for size in input_sizes:
    cities = generate_worst_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    worst_case_times.append(execution_time)

# Generate best-case execution times
best_case_times = []
for size in input_sizes:
    cities = generate_best_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    best_case_times.append(execution_time)

# Generate average-case execution times
average_case_times = []
for size in input_sizes:
    cities = generate_average_case_input(size)
    min_distance, min_path, execution_time = measure_execution_time(cities)
    average_case_times.append(execution_time)

# Plot the graphs
plt.plot(input_sizes, worst_case_times, label="Worst Case")
plt.plot(input_sizes, best_case_times, label="Best Case")
plt.plot(input_sizes, average_case_times, label="Average Case")
plt.xlabel("Number of Cities")
plt.ylabel("Execution Time")
plt.legend()
plt.show()


# In[ ]:




