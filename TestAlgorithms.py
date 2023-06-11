#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy as np
import time

class TravelingSalesmanTwoOpt:
    def __init__(self, cities):
        self.cities = cities
        self.num_cities = len(cities)
        self.tour = random.sample(cities, self.num_cities)

    def get_distance(self, city1, city2):
        return np.linalg.norm(np.array(city1) - np.array(city2))

    def improve_tour(self):
        improved = True
        while improved:
            improved = False
            for i in range(self.num_cities - 1):
                for j in range(i + 2, self.num_cities):
                    current_distance = self.get_distance(self.tour[i], self.tour[i + 1]) + self.get_distance(self.tour[j], self.tour[(j + 1) % self.num_cities])
                    new_distance = self.get_distance(self.tour[i], self.tour[j]) + self.get_distance(self.tour[i + 1], self.tour[(j + 1) % self.num_cities])
                    if new_distance < current_distance:
                        self.tour[i + 1:j + 1] = reversed(self.tour[i + 1:j + 1])
                        improved = True

    def solve(self):
        start_time = time.time()

        self.improve_tour()
        total_distance = sum(self.get_distance(self.tour[i], self.tour[(i + 1) % self.num_cities]) for i in range(self.num_cities))

        end_time = time.time()
        execution_time = end_time - start_time

        return total_distance, self.tour, execution_time


# In[2]:


import numpy as np
import random
import time

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


# In[7]:


# Input 5
cities = [(10, 10), (20, 30), (55, 25), (70, 48), (40, 60)]

#Two-opt
tsp_two_opt = TravelingSalesmanTwoOpt(cities)
total_distance, improved_tour, execution_time = tsp_two_opt.solve()

print("Two opt")
print("Total Distance:", total_distance)
print("Improved Tour:", improved_tour)
print("Execution Time:", execution_time)
print("_____________________________________________________________________________________________")
#nearest neighbour
tsp = TravelingSalesmanNearestNeighbor(cities)

start_time = time.time()

distance, path = tsp.solve()

end_time = time.time()
execution_time = end_time - start_time
print("Nearest neighbour")
print("Total Distance:", distance)
print("Path:", path)
print("Execution Time:", execution_time)


# In[8]:


# Input 8
cities = [(45, 72), (18, 30), (65, 50), (32, 88), (57, 42), (71, 23), (94, 60), (10, 80)]

#Two-opt
tsp_two_opt = TravelingSalesmanTwoOpt(cities)
total_distance, improved_tour, execution_time = tsp_two_opt.solve()

print("Two opt")
print("Total Distance:", total_distance)
print("Improved Tour:", improved_tour)
print("Execution Time:", execution_time)
print("_____________________________________________________________________________________________")
#nearest neighbour
tsp = TravelingSalesmanNearestNeighbor(cities)

start_time = time.time()

distance, path = tsp.solve()

end_time = time.time()
execution_time = end_time - start_time
print("Nearest neighbour")
print("Total Distance:", distance)
print("Path:", path)
print("Execution Time:", execution_time)


# In[6]:


# Input 50
cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(50)]

#Two-opt
tsp_two_opt = TravelingSalesmanTwoOpt(cities)
total_distance, improved_tour, execution_time = tsp_two_opt.solve()

print("Two opt")
print("Total Distance:", total_distance)
print("Improved Tour:", improved_tour)
print("Execution Time:", execution_time)
print("_____________________________________________________________________________________________")

#nearest neighbour
tsp = TravelingSalesmanNearestNeighbor(cities)

start_time = time.time()

distance, path = tsp.solve()

end_time = time.time()
execution_time = end_time - start_time
print("Nearest neighbour")
print("Total Distance:", distance)
print("Path:", path)
print("Execution Time:", execution_time)


# In[9]:


# Input 100
cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(100)]

#Two-opt
tsp_two_opt = TravelingSalesmanTwoOpt(cities)
total_distance, improved_tour, execution_time = tsp_two_opt.solve()

print("Two opt")
print("Total Distance:", total_distance)
print("Improved Tour:", improved_tour)
print("Execution Time:", execution_time)
print("_____________________________________________________________________________________________")

#nearest neighbour
tsp = TravelingSalesmanNearestNeighbor(cities)

start_time = time.time()

distance, path = tsp.solve()

end_time = time.time()
execution_time = end_time - start_time
print("Nearest neighbour")
print("Total Distance:", distance)
print("Path:", path)
print("Execution Time:", execution_time)


# In[10]:


# Input 200
cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(200)]

#Two-opt
tsp_two_opt = TravelingSalesmanTwoOpt(cities)
total_distance, improved_tour, execution_time = tsp_two_opt.solve()

print("Two opt")
print("Total Distance:", total_distance)
print("Improved Tour:", improved_tour)
print("Execution Time:", execution_time)
print("_____________________________________________________________________________________________")

#nearest neighbour
tsp = TravelingSalesmanNearestNeighbor(cities)

start_time = time.time()

distance, path = tsp.solve()

end_time = time.time()
execution_time = end_time - start_time
print("Nearest neighbour")
print("Total Distance:", distance)
print("Path:", path)
print("Execution Time:", execution_time)


# In[11]:


# Input 300
cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(300)]

#Two-opt
tsp_two_opt = TravelingSalesmanTwoOpt(cities)
total_distance, improved_tour, execution_time = tsp_two_opt.solve()

print("Two opt")
print("Total Distance:", total_distance)
print("Improved Tour:", improved_tour)
print("Execution Time:", execution_time)
print("_____________________________________________________________________________________________")

#nearest neighbour
tsp = TravelingSalesmanNearestNeighbor(cities)

start_time = time.time()

distance, path = tsp.solve()

end_time = time.time()
execution_time = end_time - start_time
print("Nearest neighbour")
print("Total Distance:", distance)
print("Path:", path)
print("Execution Time:", execution_time)


# In[12]:


# Input 400
cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(400)]

#Two-opt
tsp_two_opt = TravelingSalesmanTwoOpt(cities)
total_distance, improved_tour, execution_time = tsp_two_opt.solve()

print("Two opt")
print("Total Distance:", total_distance)
print("Improved Tour:", improved_tour)
print("Execution Time:", execution_time)
print("_____________________________________________________________________________________________")

#nearest neighbour
tsp = TravelingSalesmanNearestNeighbor(cities)

start_time = time.time()

distance, path = tsp.solve()

end_time = time.time()
execution_time = end_time - start_time
print("Nearest neighbour")
print("Total Distance:", distance)
print("Path:", path)
print("Execution Time:", execution_time)


# In[13]:


# Input 500
cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(500)]

#Two-opt
tsp_two_opt = TravelingSalesmanTwoOpt(cities)
total_distance, improved_tour, execution_time = tsp_two_opt.solve()

print("Two opt")
print("Total Distance:", total_distance)
print("Improved Tour:", improved_tour)
print("Execution Time:", execution_time)
print("_____________________________________________________________________________________________")

#nearest neighbour
tsp = TravelingSalesmanNearestNeighbor(cities)

start_time = time.time()

distance, path = tsp.solve()

end_time = time.time()
execution_time = end_time - start_time
print("Nearest neighbour")
print("Total Distance:", distance)
print("Path:", path)
print("Execution Time:", execution_time)


# In[ ]:




