import matplotlib.pyplot as plt
import heapq as hq
import osmnx as ox
import math
import time
import pandas as pd
import tracemalloc
import networkx as nx
import sklearn


#define a road class with its attributes
class Road:
    def __init__(self, name, highway = False, congested = False,  well_lit = True, passes_through = []):
        self.name = name
        self.highway = highway
        self.congested = congested
        self.well_lit = well_lit
        self.passes_through = passes_through

#avoid highways constraint
def avoid_highways(road):
    return not road.highway

#avoid congested roads constraint
def avoid_congestion(road):
    return not road.congested

#prefer well-lit roads constraint
def prefer_well_lit(graph_edge):
    for road in roads:
        if graph_edge.get('name', '') == road.name:
            print("Checking if ", road.name, " is well lit.")
            return road.well_lit
    return False

#avoid unsafe areas constraint
def avoid_unsafe_areas(road):
    for area in road.passes_through:
        if area in unsafe_areas:
            return False
    return True


#defines some roads around oxford
roads = [
    Road("St Clements Street", highway = False, congested = True, well_lit = True, passes_through = ["St Clements"]),
    Road("Headington Road", highway = False, congested = False, well_lit = True, passes_through = ["Headington", "St Clements"]),
    Road("Cherwell Drive", highway = False, congested = True, well_lit = False, passes_through = ["Marston"]),
    Road("Oxford Road", highway = True, congested = False, well_lit = False, passes_through = ["Woodstock"]),
    Road("Banbury Road", highway = False, congested = False, well_lit = True, passes_through = ["Central Oxford", "Summertown"]),
    Road("Chadlington Road", highway = False, congested = False, well_lit = False, passes_through = ["Chadlington"]),
]

#defines some areas around oxford
areas = ["Headington", "Central Oxford", "St Clements", "Summertown", "Marston", "Woodstock", "Chadlington"]

#defines some unsafe areas
unsafe_areas = ["Chadlington"]

scenic_areas = ["Marston"]

#defines the hard constraints
hard_constraints = [
    avoid_highways,
    avoid_congestion,
    lambda road: avoid_unsafe_areas(road)
]

#combined logic for the constraints
def include(graph_edge):
    for road in roads:
        if graph_edge.get('name', '') == road.name:
            print("Checking, ", road.name)
            #checks if the road passes through a scenic area
            for area in road.passes_through:
                if area in scenic_areas:
                    return True
            for constraint in hard_constraints:
                if not constraint(road):
                    return False
    return True

# adjust pandas display settings
pd.set_option('display.max_rows', None)  # display all rows
pd.set_option('display.max_columns', None)  # display all columns

#coordinates of Headington Gipsy Lane Campus and Diddly Squat Farm
start_point = (51.755318, -1.225519) #Oxford Brookes
end_point = (51.916397, -1.541773) #Diddly Squat

#find the midpoint between Oxford Brookes and Diddly Squat
mid_point = ((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2)

#get the road network within 13km of the midpoint between the start and end
G = ox.graph_from_point(mid_point, dist=13000, network_type="drive")

#find the nodes nearest to start, mid and end points
start_node = ox.distance.nearest_nodes(G, X=start_point[1], Y=start_point[0])
mid_node = ox.distance.nearest_nodes(G, X=mid_point[1], Y=mid_point[0])
end_node = ox.distance.nearest_nodes(G, X=end_point[1], Y=end_point[0])

#plot the graph with start and end nodes
fig, ax = ox.plot_graph(G, node_size = 20, edge_linewidth=1, show=False, close=False)
plt.scatter([G.nodes[start_node]['x']], [G.nodes[start_node]['y']], c="green", s=150, zorder=5, label="Start Point")
plt.scatter([G.nodes[end_node]['x']],[G.nodes[end_node]['y']], c="red", s=150, zorder=5, label="End Point")
plt.title("Road Network between Oxford Brookes and Diddly Squat")

#add a legend for stand and end
plt.legend(loc="upper right")

#save plot as an image
fig.set_size_inches(50, 50)
plt.savefig("road_network_OBU_DS.png", dpi=300, bbox_inches='tight')

#display the plot
plt.show()

#definition of Euclidian heuristic function
def euclidian_distance(n1, n2, G):
    x1, y1 = G.nodes[n1]['x'], G.nodes[n1]['y']
    x2, y2 = G.nodes[n2]['x'], G.nodes[n2]['y']
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

#A* Algorithm using Euclidian Heuristics
def a_star(G, start_node, end_node):
    #track number of explored nodes
    explored_nodes = 0

    #start timing
    start_time = time.time()

    #track memory allocation
    tracemalloc.start()

    #priority queue for open set
    open_set = []
    hq.heappush(open_set, (0, start_node))

    #path tracking and g_scores
    came_from = {start_node: None}
    g_score = {node: float('inf') for node in G.nodes}
    g_score[start_node] = 0

    #runs the A* search
    while open_set:
        #pop the node with lowest f_score
        current_f, current = hq.heappop(open_set)
        explored_nodes += 1

        #if reached end_node then reconstruct the path
        if current == end_node:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            execution_time = time.time() - start_time
            peak_memory = tracemalloc.get_traced_memory()[1] # Get peak memory usage
            tracemalloc.stop()
            return path[::-1], g_score[end_node], explored_nodes, execution_time, peak_memory #reverses the path to show from start to end

        #explore neighbors of the current node
        for neighbour in G[current]:
            #get edge length to neighbour
            cost = G.edges[current, neighbour, 0].get('length')
            tentative_g_score = g_score[current] + cost

            if neighbour not in g_score or tentative_g_score < g_score[neighbour]:
                g_score[neighbour] = tentative_g_score
                f_score = tentative_g_score + euclidian_distance(neighbour, end_node, G)
                hq.heappush(open_set, (f_score, neighbour))
                came_from[neighbour] = current

    peak_memory = tracemalloc.get_traced_memory()[1] # Get peak memory usage
    tracemalloc.stop()
    return None, float("inf"), explored_nodes, time.time() - start_time, peak_memory

#A* Algorithm using Euclidian Heuristics
def logical_a_star(G, start_node, end_node):
    #track number of explored nodes
    explored_nodes = 0

    #start timing
    start_time = time.time()

    #track memory allocation
    tracemalloc.start()

    #priority queue for open set
    open_set = []
    hq.heappush(open_set, (0, start_node))

    #path tracking and g_scores
    came_from = {start_node: None}
    g_score = {node: float('inf') for node in G.nodes}
    g_score[start_node] = 0

    #runs the A* search
    while open_set:
        #pop the node with lowest f_score
        current_f, current = hq.heappop(open_set)
        explored_nodes += 1

        #if reached end_node then reconstruct the path
        if current == end_node:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            execution_time = time.time() - start_time
            peak_memory = tracemalloc.get_traced_memory()[1] # Get peak memory usage
            tracemalloc.stop()
            return path[::-1], g_score[end_node], explored_nodes, execution_time, peak_memory #reverses the path to show from start to end

        #explore neighbors of the current node
        for neighbour in G[current]:
            #get edge length to neighbour
            cost = G.edges[current, neighbour, 0].get('length')

            if include(G.edges[current, neighbour, 0]):
                well_lit_weight = 0.0 if prefer_well_lit(G.edges[current, neighbour, 0]) else 10.0

                tentative_g_score = g_score[current] + cost


                if neighbour not in g_score or tentative_g_score < g_score[neighbour]:
                    g_score[neighbour] = tentative_g_score
                    f_score = tentative_g_score + euclidian_distance(neighbour, end_node, G) + well_lit_weight
                    hq.heappush(open_set, (f_score, neighbour))
                    came_from[neighbour] = current

    peak_memory = tracemalloc.get_traced_memory()[1] # Get peak memory usage
    tracemalloc.stop()
    return None, float("inf"), explored_nodes, time.time() - start_time, peak_memory

def backtracking_algorithm(G, current, end_node, visited = None, path = None, explored_nodes = 0, start_time = time.time()):
    #track memory allocation
    tracemalloc.start()

    if visited is None:
        visited = set()
    if path is None:
        path = [current]

    visited.add(current)

    #if the end has been reached, return the path
    if current == end_node:
        peak_memory = tracemalloc.get_traced_memory()[1]  # Get peak memory usage
        tracemalloc.stop()

        # calculate total cost
        total_cost = 0
        for i in range(len(path) - 1):
            total_cost += G.edges[path[i], path[i + 1], 0].get('length')
        return path, total_cost, explored_nodes, time.time() - start_time, peak_memory

    neighbours = []
    #explore non well-lit roads
    for neighbour in G.neighbors(current):
        if include(G.edges[current, neighbour, 0]) and neighbour not in visited:
            explored_nodes += 1
            neighbours.append(neighbour)

    for neighbour in neighbours:
        if prefer_well_lit(G.edges[current, neighbour, 0]):
            new_path = backtracking_algorithm(G, neighbour, end_node, visited, path + [neighbour], explored_nodes = explored_nodes, start_time = start_time)
        else:
            new_path = backtracking_algorithm(G, neighbour, end_node, visited, path + [neighbour], explored_nodes = explored_nodes, start_time = start_time)
        if new_path:
            return new_path

    #backtrack
    return None

#function to run the algorithms, then stores all the returned data in a dictionary
def compare_algorithms(G, start_node, end_node):
    results = {}
    algorithms = {
        "Logical A Star": logical_a_star,
        "Backtracking Algorithm": backtracking_algorithm,
        "A Star": a_star
    }

    for algo_name, algo_func in algorithms.items():
        path, cost, explored_nodes, time_taken, peak_memory = algo_func(G, start_node, end_node)
        optimality_ratio = cost / nx.shortest_path_length(G, start_node, end_node, weight='length')
        fig, ax = ox.plot_graph_route(G, path, route_linewidth=2, node_size=0)
        results[algo_name] = {
            "Path Length (meters)": cost,
            "Explored nodes": explored_nodes,
            "Time taken (s)": time_taken,
            "Peak Memory Usage": peak_memory,
            "Optimality Ratio": optimality_ratio
        }

    return results

#calls compare_algorithms, converts results into a dataframe
results = compare_algorithms(G, start_node, end_node)
df_results = pd.DataFrame(results).T
print(df_results)

#extracts specific metrics for graph creation and analysis
algorithms = df_results.index
path_length = df_results["Path Length (meters)"]
explored_nodes = df_results["Explored nodes"]
time_taken = df_results["Time taken (s)"]
peak_memory = df_results["Peak Memory Usage"]
optimality_ratio = df_results["Optimality Ratio"]

#creates graphs for all the metrics
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
axes = axes.flatten()

metrics = [
    (path_length, "Path Length (meters)", "Path Length"),
    (explored_nodes, "Explored nodes", "Explored nodes"),
    (time_taken, "Time taken (s)", "Time taken"),
    (peak_memory, "Peak Memory Usage", "Peak Memory Usage"),
    (optimality_ratio, "Optimality Ratio", "Optimality Ratio")
]

colors = ["red", "green", "blue"]

for i, (data, ylabel, title) in enumerate(metrics):
    ax = axes[i]
    ax.bar(algorithms, data, color=colors)
    ax.set_title(title)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=45)

axes[-1].axis("off")

plt.tight_layout()
plt.show()

#calculating normalised scores for the algorithms, with weighting to assign priority to path length
scores = {
    "Logical A Star": 0.0,
    "Backtracking Algorithm": 0.0,
    "A Star": 0.0
}

#assigning weights to the metrics
weights = {
    "path_length": 0.3,
    "explored_nodes": 0.1333,
    "time_taken": 0.1333,
    "peak_memory": 0.1333,
    "optimality_ratio": 0.3
}

#getting best and worst values for the metrics
best_values = {
    "path_length": path_length.min(),
    "explored_nodes": explored_nodes.min(),
    "time_taken": time_taken.min(),
    "peak_memory": peak_memory.min(),
    "optimality_ratio": optimality_ratio.min()
}

worst_values = {
    "path_length": path_length.max(),
    "explored_nodes": explored_nodes.max(),
    "time_taken": time_taken.max(),
    "peak_memory": peak_memory.max(),
    "optimality_ratio": optimality_ratio.max()
}

#generates the total score for each algorithm
for algorithm in algorithms:
    #path length score
    scores[algorithm] += (path_length[algorithm] - best_values["path_length"]) / (worst_values["path_length"] - best_values["path_length"]) * weights["path_length"]

    #explored nodes score
    scores[algorithm] += (explored_nodes[algorithm] - best_values["explored_nodes"]) / (worst_values["explored_nodes"] - best_values["explored_nodes"]) * weights["explored_nodes"]

    #time taken score
    scores[algorithm] += (time_taken[algorithm] - best_values["time_taken"]) / (worst_values["time_taken"] - best_values["time_taken"]) * weights["time_taken"]

    #peak memory score
    scores[algorithm] += (peak_memory[algorithm] - best_values["peak_memory"]) / (worst_values["peak_memory"] - best_values["peak_memory"]) * weights["peak_memory"]

    #optimality ratio score
    scores[algorithm] += (optimality_ratio[algorithm] - best_values["optimality_ratio"]) / (worst_values["optimality_ratio"] - best_values["optimality_ratio"]) * weights["optimality_ratio"]

df_scores = pd.DataFrame(list(scores.items()), columns=["Algorithm", "Score (lower is better)"]).set_index("Algorithm")

#plot the table
fig, ax = plt.subplots(figsize=(20, 2))
ax.axis('off')

#create the table
table = ax.table(cellText=df_scores.values, colLabels=df_scores.columns, rowLabels=df_scores.index, cellLoc='center', loc='center')

plt.show()