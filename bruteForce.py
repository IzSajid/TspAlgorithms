#!/usr/bin/env python
# coding: utf-8

# In[2]:


import itertools

class TravelingSalesmanBruteForce:
    def __init__(self, cities):
        self.cities = cities
        self.num_cities = len(cities)

    def get_distance(self, city1, city2):
        x1, y1 = city1
        x2, y2 = city2
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def solve(self):
        min_distance = float('inf')
        min_path = None

        for path in itertools.permutations(range(self.num_cities)):
            distance = 0
            for i in range(self.num_cities - 1):
                distance += self.get_distance(self.cities[path[i]], self.cities[path[i + 1]])
            distance += self.get_distance(self.cities[path[-1]], self.cities[path[0]])

            if distance < min_distance:
                min_distance = distance
                min_path = path

        return min_distance, [self.cities[i] for i in min_path]


# In[5]:


import time

# Function to measure the execution time of your algorithm
def measure_time_complexity():
    start_time = time.time()

    # Your algorithm code here
    cities = [(10, 10), (20, 30), (55, 25), (70, 48), (40, 60)]
    tsp_bf = TravelingSalesmanBruteForce(cities)
    min_distance, min_path = tsp_bf.solve()

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time)
    print("Minimum distance:", min_distance)
    print("Minimum path:", min_path)

# Call the function to measure the time complexity
measure_time_complexity()


# In[3]:


import time

# Function to measure the execution time of your algorithm
def measure_time_complexity():
    start_time = time.time()

    # Your algorithm code here
    cities = [(45, 72), (18, 30), (65, 50), (32, 88), (57, 42), (71, 23), (94, 60), (10, 80)]
    tsp_bf = TravelingSalesmanBruteForce(cities)
    min_distance, min_path = tsp_bf.solve()

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time)
    print("Minimum distance:", min_distance)
    print("Minimum path:", min_path)

# Call the function to measure the time complexity
measure_time_complexity()


# In[10]:


import random
import time

def generate_random_cities(num_cities):
    cities = []
    for _ in range(num_cities):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        cities.append((x, y))
    return cities

def test_brute_force(cities):
    tsp_bf = TravelingSalesmanBruteForce(cities)

    start_time = time.time()
    min_distance_bf, min_path_bf = tsp_bf.solve()
    end_time = time.time()
    execution_time = end_time - start_time

    # Print the result brute force
    print("Using brute force:")
    print("Minimum distance:", min_distance_bf)
    print("Minimum path:", min_path_bf)
    print("Execution time:", execution_time)

def main():
    # User input for the number of cities
    num_cities = int(input("Enter the number of cities: "))

    # Generate random cities
    cities = generate_random_cities(num_cities)

    # Print the cities
    print("Cities:", cities)

    # Test brute force and measure time complexity
    test_brute_force(cities)

if __name__ == '__main__':
    main()


# In[25]:


import matplotlib.pyplot as plt
import time
import random
import itertools
import math

# Generate worst-case input
def generate_worst_case_input(size):
    cities = [(i, 0) for i in range(size)]
    return cities

# Generate best-case input
def generate_best_case_input(size):
    angle = 2 * math.pi / size
    cities = [(math.cos(i * angle), math.sin(i * angle)) for i in range(size)]
    return cities

# Generate average-case input
def generate_average_case_input(size):
    cities = [(random.random(), random.random()) for _ in range(size)]
    return cities

# Measure execution time for a given input size
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanBruteForce(cities)
    start_time = time.time()
    min_distance, min_path = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, min_path, execution_time

# Generate input sizes (number of cities)
input_sizes = [3, 4, 5 , 6 , 7]

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


# In[30]:


import matplotlib.pyplot as plt
import time
import random
import itertools
import math

# Generate worst-case input
def generate_worst_case_input(size):
    cities = [(i, 0) for i in range(size)]
    return cities

# Generate best-case input
def generate_best_case_input(size):
    angle = 2 * math.pi / size
    cities = [(math.cos(i * angle), math.sin(i * angle)) for i in range(size)]
    return cities

# Generate average-case input
def generate_average_case_input(size):
    cities = [(random.random(), random.random()) for _ in range(size)]
    return cities

# Measure execution time for a given input size
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanBruteForce(cities)
    start_time = time.time()
    min_distance, min_path = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, min_path, execution_time

# Generate input sizes (number of cities)
input_sizes = [4, 5 , 6 , 7 , 8]

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


import matplotlib.pyplot as plt
import time
import random
import itertools
import math

# Generate worst-case input
def generate_worst_case_input(size):
    cities = [(i, 0) for i in range(size)]
    return cities

# Generate best-case input
def generate_best_case_input(size):
    angle = 2 * math.pi / size
    cities = [(math.cos(i * angle), math.sin(i * angle)) for i in range(size)]
    return cities

# Generate average-case input
def generate_average_case_input(size):
    cities = [(random.random(), random.random()) for _ in range(size)]
    return cities

# Measure execution time for a given input size
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanBruteForce(cities)
    start_time = time.time()
    min_distance, min_path = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, min_path, execution_time

# Generate input sizes (number of cities)
input_sizes = [5 , 6 , 7 , 8 , 9]

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


import matplotlib.pyplot as plt
import time
import random
import itertools
import math

# Generate worst-case input
def generate_worst_case_input(size):
    cities = [(i, 0) for i in range(size)]
    return cities

# Generate best-case input
def generate_best_case_input(size):
    angle = 2 * math.pi / size
    cities = [(math.cos(i * angle), math.sin(i * angle)) for i in range(size)]
    return cities

# Generate average-case input
def generate_average_case_input(size):
    cities = [(random.random(), random.random()) for _ in range(size)]
    return cities

# Measure execution time for a given input size
def measure_execution_time(cities):
    tsp_solver = TravelingSalesmanBruteForce(cities)
    start_time = time.time()
    min_distance, min_path = tsp_solver.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, min_path, execution_time

# Generate input sizes (number of cities)
input_sizes = [6 , 7, 8, 9, 10]

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




