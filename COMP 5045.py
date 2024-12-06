import matplotlib.pyplot as plt
import heapq as hq
from collections import deque
import osmnx as ox
import math
import time
import pandas as pd
import tracemalloc
import networkx as nx

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

#Bidirectional Algortihm
def bfs_step(queue, visited, parent, other_visited):
    #performs one step of BFS in one direction
    current = queue.popleft()
    for neighbour in G.neighbors(current):
        if neighbour not in visited:
            queue.append(neighbour)
            visited[neighbour] = True
            parent[neighbour] = current
            #checks if node has been visited in other direction
            if neighbour in other_visited:
                return neighbour
    return None

def construct_path(common_node, start_parent, end_parent):
    #constructs path from start to end
    path = []
    #trace back from common node to start
    node = common_node
    while node is not None:
        path.append(node)
        node = start_parent.get(node)
    path.reverse()

    #trace back from common node to end
    node = end_parent.get(common_node)
    while node is not None:
        path.append(node)
        node = end_parent.get(node)

    #calculate total cost
    total_cost = 0
    for i in range(len(path) - 1):
        total_cost += G.edges[path[i], path[i+1], 0].get('length')

    return path, total_cost

def bidirectional_bfs(G, start_node, end_node):
    #performs bidirectional BFS to find the path between start_node and end_node

    #tracks number of explored nodes
    explored_nodes = 0

    #start timing
    start_time = time.time()

    #track memory allocation
    tracemalloc.start()

    #intialise queues and visited and parent dictionaries for both directions
    start_queue = deque([start_node])
    end_queue = deque([end_node])
    start_visited = {start_node: True}
    end_visited = {end_node: True}
    start_parent = {start_node: None}
    end_parent = {end_node: None}

    while start_queue and end_queue:
        #step forward in start direction
        common_node = bfs_step(start_queue, start_visited, start_parent, end_visited)
        explored_nodes += 1
        if common_node:
            path, total_cost = construct_path(common_node, start_parent, end_parent)
            execution_time = time.time() - start_time
            peak_memory = tracemalloc.get_traced_memory()[1] # Get peak memory usage
            tracemalloc.stop()
            return path, total_cost, explored_nodes, execution_time, peak_memory

        #step forward in end direction
        common_node = bfs_step(end_queue, end_visited, end_parent, start_visited)
        explored_nodes += 1
        if common_node:
            path, total_cost = construct_path(common_node, start_parent, end_parent)
            execution_time = time.time() - start_time
            peak_memory = tracemalloc.get_traced_memory()[1] # Get peak memory usage
            tracemalloc.stop()
            return path, total_cost, explored_nodes, execution_time, peak_memory

    peak_memory = tracemalloc.get_traced_memory()[1] # Get peak memory usage
    tracemalloc.stop()
    return None, float("inf"), explored_nodes, time.time() - start_time, peak_memory

#Dijkstra's Algorithm
def dijkstra(G, start_node, end_node):
    # track number of explored nodes
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

    while open_set:
        current_cost, current = hq.heappop(open_set)
        explored_nodes += 1

        #if reached end_node then reconstruct path
        if current == end_node:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            execution_time = time.time() - start_time
            peak_memory = tracemalloc.get_traced_memory()[1] # Get peak memory usage
            tracemalloc.stop()
            return path[::-1], g_score[end_node], explored_nodes, execution_time, peak_memory

        #explore neighbors of the current node
        for neighbour in G[current]:
            #get edge length to neighbour
            edge_cost = G.edges[current, neighbour, 0].get('length')
            tentative_cost = g_score[current] + edge_cost

            if neighbour not in g_score or tentative_cost < g_score[neighbour]:
                g_score[neighbour] = tentative_cost
                hq.heappush(open_set, (tentative_cost, neighbour))
                came_from[neighbour] = current

    peak_memory = tracemalloc.get_traced_memory()[1] # Get peak memory usage
    tracemalloc.stop()
    return None, float("inf"), explored_nodes, time.time() - start_time, peak_memory

#optimisation 1 - prune graph
def prune_graph(G):
    #calculate the euclidian distance between start and midpoint, add 100m, assign this as the threshold
    xM, yM = G.nodes[mid_node]['x'], G.nodes[mid_node]['y']
    x1, y1 = G.nodes[start_node]['x'], G.nodes[start_node]['y']
    threshold = (math.sqrt((xM - x1) ** 2 + (yM - y1) ** 2) + 100)

    #remove nodes that farther than the threshold from the midpoint
    nodes_to_remove = []
    for node in G.nodes:
        x, y = G.nodes[node]['x'], G.nodes[node]['y']
        #calculate euclidian distance between midpoint and start
        distance_to_midpoint = math.sqrt((xM - x) ** 2 + (yM - y) ** 2)

        if distance_to_midpoint > threshold:
            nodes_to_remove.append(node)

    #removes the nodes
    G.remove_nodes_from(nodes_to_remove)

    return G

#optimisation 2 - remove low degree nodes
def remove_low_degree_nodes(G, min_degree = 3):
    nodes_to_remove = []
    for node in G.nodes:
        if G.degree(node) < min_degree:
            nodes_to_remove.append(node)
    G.remove_nodes_from(nodes_to_remove)

    return G

#definition of Euclidian heuristic function
def euclidian_distance_optimised(n1, n2, g):
    x1, y1 = g.nodes[n1]['x'], g.nodes[n1]['y']
    x2, y2 = g.nodes[n2]['x'], g.nodes[n2]['y']
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

#A* Algorithm using Euclidian Heuristics
def a_star_optimised(G, start_node, end_node):
    #copies the graph so no changes are made to the original
    g = G.copy()

    #track number of explored nodes
    explored_nodes = 0

    #start timing
    start_time = time.time()

    #track memory allocation
    tracemalloc.start()

    #optimisations
    g = prune_graph(g)
    g = remove_low_degree_nodes(g)

    #priority queue for open set
    open_set = []
    hq.heappush(open_set, (0, start_node))

    #path tracking and g_scores
    came_from = {start_node: None}
    g_score = {node: float('inf') for node in g.nodes}
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
        for neighbour in g[current]:
            #get edge length to neighbour
            cost = g.edges[current, neighbour, 0].get('length')
            tentative_g_score = g_score[current] + cost

            if neighbour not in g_score or tentative_g_score < g_score[neighbour]:
                g_score[neighbour] = tentative_g_score
                f_score = tentative_g_score + euclidian_distance_optimised(neighbour, end_node, g)
                hq.heappush(open_set, (f_score, neighbour))
                came_from[neighbour] = current

    peak_memory = tracemalloc.get_traced_memory()[1] # Get peak memory usage
    tracemalloc.stop()
    return None, float("inf"), explored_nodes, time.time() - start_time, peak_memory

#Bidirectional Algortihm
def bfs_step_optimised(g, queue, visited, parent, other_visited):
    #performs one step of BFS in one direction
    current = queue.popleft()
    for neighbour in g.neighbors(current):
        if neighbour not in visited:
            queue.append(neighbour)
            visited[neighbour] = True
            parent[neighbour] = current
            #checks if node has been visited in other direction
            if neighbour in other_visited:
                return neighbour
    return None

def construct_path_optimised(g, common_node, start_parent, end_parent):
    #constructs path from start to end
    path = []
    #trace back from common node to start
    node = common_node
    while node is not None:
        path.append(node)
        node = start_parent.get(node)
    path.reverse()

    #trace back from common node to end
    node = end_parent.get(common_node)
    while node is not None:
        path.append(node)
        node = end_parent.get(node)

    #calculate total cost
    total_cost = 0
    for i in range(len(path) - 1):
        total_cost += g.edges[path[i], path[i+1], 0].get('length')

    return path, total_cost

def bidirectional_bfs_optimised(G, start_node, end_node):
    #copies the graph so no changes are made to the original
    g = G.copy()

    #tracks number of explored nodes
    explored_nodes = 0

    #start timing
    start_time = time.time()

    #track memory allocation
    tracemalloc.start()

    #optimisations
    g = prune_graph(g)
    g = remove_low_degree_nodes(g)

    #intialise queues and visited and parent dictionaries for both directions
    start_queue = deque([start_node])
    end_queue = deque([end_node])
    start_visited = {start_node: True}
    end_visited = {end_node: True}
    start_parent = {start_node: None}
    end_parent = {end_node: None}

    while start_queue and end_queue:
        #step forward in start direction
        common_node = bfs_step_optimised(g, start_queue, start_visited, start_parent, end_visited)
        explored_nodes += 1
        if common_node:
            path, total_cost = construct_path_optimised(g, common_node, start_parent, end_parent)
            execution_time = time.time() - start_time
            peak_memory = tracemalloc.get_traced_memory()[1] # Get peak memory usage
            tracemalloc.stop()
            return path, total_cost, explored_nodes, execution_time, peak_memory

        #step forward in end direction
        common_node = bfs_step_optimised(g, end_queue, end_visited, end_parent, start_visited)
        explored_nodes += 1
        if common_node:
            path, total_cost = construct_path_optimised(g, common_node, start_parent, end_parent)
            execution_time = time.time() - start_time
            peak_memory = tracemalloc.get_traced_memory()[1] # Get peak memory usage
            tracemalloc.stop()
            return path, total_cost, explored_nodes, execution_time, peak_memory

    peak_memory = tracemalloc.get_traced_memory()[1] # Get peak memory usage
    tracemalloc.stop()
    return None, float("inf"), explored_nodes, time.time() - start_time, peak_memory

#Dijkstra's Algorithm
def dijkstra_optimised(G, start_node, end_node):
    #copies the graph so no changes are made to the original
    g = G.copy()

    # track number of explored nodes
    explored_nodes = 0

    #start timing
    start_time = time.time()

    #track memory allocation
    tracemalloc.start()

    #optimisations
    g = prune_graph(g)
    g = remove_low_degree_nodes(g)

    #priority queue for open set
    open_set = []
    hq.heappush(open_set, (0, start_node))

    #path tracking and g_scores
    came_from = {start_node: None}
    g_score = {node: float('inf') for node in g.nodes}
    g_score[start_node] = 0

    while open_set:
        current_cost, current = hq.heappop(open_set)
        explored_nodes += 1

        #if reached end_node then reconstruct path
        if current == end_node:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            execution_time = time.time() - start_time
            peak_memory = tracemalloc.get_traced_memory()[1] # Get peak memory usage
            tracemalloc.stop()
            return path[::-1], g_score[end_node], explored_nodes, execution_time, peak_memory

        #explore neighbors of the current node
        for neighbour in g[current]:
            #get edge length to neighbour
            edge_cost = g.edges[current, neighbour, 0].get('length')
            tentative_cost = g_score[current] + edge_cost

            if neighbour not in g_score or tentative_cost < g_score[neighbour]:
                g_score[neighbour] = tentative_cost
                hq.heappush(open_set, (tentative_cost, neighbour))
                came_from[neighbour] = current

    peak_memory = tracemalloc.get_traced_memory()[1] # Get peak memory usage
    tracemalloc.stop()
    return None, float("inf"), explored_nodes, time.time() - start_time, peak_memory

#function to run the algorithms, then stores all the returned data in a dictionary
def compare_algorithms(G, start_node, end_node):
    results = {}
    algorithms = {
        "A Star": a_star,
        "Bidirectional BFS": bidirectional_bfs,
        "Dijkstra": dijkstra,
        "A Star Optimised": a_star_optimised,
        "Bidirectional BFS Optimised": bidirectional_bfs_optimised,
        "Dijkstra Optimised": dijkstra_optimised,
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
    "A Star": 0.0,
    "Bidirectional BFS": 0.0,
    "Dijkstra": 0.0,
    "A Star Optimised" : 0.0,
    "Bidirectional BFS Optimised" : 0.0,
    "Dijkstra Optimised" : 0.0,
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