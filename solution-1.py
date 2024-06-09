# Removing 0 deg nodes from Set A and Set B + ILP

from ortools.linear_solver import pywraplp
import sys
# import time

class SegmentTree:
    def __init__(self, size, arr):
        self.n = size
        self.a = arr
        self.st = [0] * (4 * size)
        self.construct(0, 0, size - 1)

    def construct(self, node, ll, rl):
        if ll == rl:
            self.st[node] = self.a[ll]
        else:
            left = self.construct(2 * node + 1, ll, (ll + rl) // 2)
            right = self.construct(2 * node + 2, (ll + rl) // 2 + 1, rl)
            self.st[node] = left + right
        return self.st[node]

    def query(self, node, ll, rl, ql, qr):
        if ll >= ql and rl <= qr:
            return self.st[node]
        elif rl < ql or ll > qr:
            return 0
        left = self.query(2 * node + 1, ll, (ll + rl) // 2, ql, qr)
        right = self.query(2 * node + 2, (ll + rl) // 2 + 1, rl, ql, qr)
        return left + right

    def update(self, node, ll, rl, q, val):
        if rl < q or ll > q:
            return self.st[node]
        if q == ll and q == rl:
            self.st[node] = val
        else:
            left = self.update(2 * node + 1, ll, (ll + rl) // 2, q, val)
            right = self.update(2 * node + 2, (ll + rl) // 2 + 1, rl, q, val)
            self.st[node] = left + right
        return self.st[node]

    def range_query(self, ql, qr):
        return self.query(0, 0, self.n - 1, ql, qr)

    def point_update(self, q, val):
        self.update(0, 0, self.n - 1, q, val)

def countcrossings_segtree(B, edges, right_order):
    B_len = len(B)
    edges.sort(key=lambda x: (x[0], right_order[x[1]]))

    arr = [0] * (B_len + 1)
    segTree = SegmentTree(B_len + 1, arr)

    crossings = 0
    for edge in edges:
        old = segTree.range_query(right_order[edge[1]], right_order[edge[1]])
        segTree.point_update(right_order[edge[1]], old + 1)

        if B_len > right_order[edge[1]] + 1:
            crossings_found = segTree.range_query(right_order[edge[1]] + 1, B_len)
            crossings += crossings_found

    return crossings

def median_barycenter(graph, n0, n1):
    v = []
    for i in range(n0 + 1, n0 + n1 + 1):
        graph[i].sort()
        neighbour = graph[i]
        length = len(neighbour)
        median = 0
        if length == 0:
            median = 0
        elif length % 2 == 0:
            median = (neighbour[(length - 2) // 2] + neighbour[length // 2]) / 2.0
        else:
            median = neighbour[(length -1) // 2]
        
        v.append((median, i))
    v.sort()
    B = [pr[1] for pr in v]
    del v

    return B

def get_order(solution, n0, n1):
    column_sums = [sum(col) for col in zip(*solution)]
    column_sums_with_indices = [(i + 1, column_sums[i]) for i in range(n1)]
    sorted_columns = sorted(column_sums_with_indices, key=lambda x: x[1], reverse=False)
    order = [col[0] + n0 for col in sorted_columns]

    # Clear intermediate variables
    del column_sums, column_sums_with_indices, sorted_columns

    return order

def solve_ilp(crossing_matrix, n1, barycenter_crossings):
    solver = pywraplp.Solver.CreateSolver('CBC')

    if not solver:
        raise Exception("Could not create solver")

    # Define the binary decision variables
    m = {}
    for i in range(1, n1 + 1):
        for j in range(1, n1 + 1):
            m[(i, j)] = solver.IntVar(0, 1, f'm_{i}_{j}')

    # Define the objective function to minimize
    solver.Minimize(solver.Sum(m[(i, j)] * crossing_matrix[i - 1][j - 1] for i in range(1, n1 + 1) for j in range(1, n1 + 1)))

    # Add constraints of barycenter crossings
    solver.Add(solver.Sum(m[(i, j)] * crossing_matrix[i - 1][j - 1] for i in range(1, n1 + 1) for j in range(1, n1 + 1)) <= barycenter_crossings)


    # Add constraints for transitivity
    for i in range(1, n1 + 1):
        for j in range(1, n1 + 1):
            for k in range(1, n1 + 1):
                if i != j and j != k and k != i:
                    solver.Add(m[(i, j)] + m[(j, k)] - m[(i, k)] <= 1)

    # Add mutual exclusion constraints
    for i in range(1, n1 + 1):
        for j in range(1, n1 + 1):
            if i != j:
                solver.Add(m[(i, j)] + m[(j, i)] == 1)

    # Solve the ILP problem
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        # Retrieve the solution
        solution = [[int(m[(i, j)].SolutionValue()) for j in range(1, n1 + 1)] for i in range(1, n1 + 1)]

        # Clear unused variables to reduce memory usage
        del m

        return solution
    else:
        raise Exception("No optimal solution found.")

def get_crossing_matrix(graph, n0, n1):
    c = [[0 for _ in range(n1)] for _ in range(n1)]

    for i in range(n1):
        for j in range(n1):
            Nvi = graph[i + n0 + 1]
            Nvj = graph[j + n0 + 1]

            crossing = 0
            l = 0  # Left index
            r = 0  # Right index

            while l < len(Nvi):
                while r < len(Nvj) and Nvi[l] > Nvj[r]:
                    r += 1
                crossing += r
                l += 1

            c[i][j] = crossing
    
    return c

def get_barycenter_crossings(graph, n0, n1):
    B = median_barycenter(graph, n0, n1) 
    right_order = {B[i]: i for i in range(len(B))}

    edges = []
    for i in range(1, n0+1):
        for u in graph[i]:
            edges.append((i, u))

    barycenter_crossings = countcrossings_segtree(B, edges, right_order)
    del edges
    return barycenter_crossings

def preprocess1(graph, deg, n0, n1):
    # remove 0 deg nodes from set A and set B
    cur = 0
    fil_graph = {}
    new_final_to_original = {}
    new_original_to_final = {}
    new_deg = {}
    zero_deg_node_b = []

    for i in range(1, n0+1):
        if deg[i] == 0:
            continue
        else:
            cur += 1
            fil_graph[cur] = graph[i]
            new_deg[cur] = len(graph[i])
            new_final_to_original[cur] = i
            new_original_to_final[i] = cur
    
    updated_n0 = cur

    for i in range(n0+1, n0+n1+1):
        if deg[i]==0:
            zero_deg_node_b.append(i)
            continue
        else:
            cur += 1
            fil_graph[cur] = []
            for u in graph[i]:
                fil_graph[cur].append(new_original_to_final[u])
            new_deg[cur] = len(graph[i])
            new_final_to_original[cur] = i
            new_original_to_final[i] = cur

    for i in range(1, updated_n0+1):
        child_list = []
        for u in fil_graph[i]:
            child_list.append(new_original_to_final[u])
        fil_graph[i] = child_list

    updated_n1 = cur - updated_n0

    return fil_graph, updated_n0, updated_n1, new_final_to_original, zero_deg_node_b

def read_graph():
    adjacency_list = {}
    complete = False
    deg = {}
    p_line = None

    edge_count = 0
    n0, n1, m = 0, 0, 0

    for line in sys.stdin:
        if line.startswith('c'):
            continue

        if p_line is None and line.startswith('p'):
            p_line = line.strip()
            _, ocr, n0, n1, m = p_line.split()
            n0 = int(n0)
            n1 = int(n1)
            m = int(m)

            if n0 * n1 == m:
                complete = True
                return adjacency_list, n0, n1, complete, deg

            adjacency_list = {i: [] for i in range(1, n0 + n1 + 1)}
            deg = {i: 0 for i in range(1, n0 + n1+ 1)}
            continue

        u, v = map(int, line.strip().split())
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
        deg[u] += 1
        deg[v] += 1
        edge_count += 1

    if edge_count != m:
        raise ValueError("Edge count does not match the specified number of edges.")

    return adjacency_list, n0, n1, complete, deg

def main():
    graph, n0, n1, complete, deg = read_graph()

    if complete:
        order = list(range(n0 + 1, n0 + n1 + 1))
        for element in order:
            print(element)
        return
    
    for node in graph:
        graph[node].sort()
    
    # Median Barycenter if max_degree <= 2; Removed
    # if max_degree <= 2:
    #     order = median_barycenter(graph, n0, n1)
    #     for element in order:
    #         print(element)
    #     return
    
    min_degree = min(deg)

    if min_degree != 0:            
        barycenter_crossing = get_barycenter_crossings(graph, n0, n1)
        crossing_matrix = get_crossing_matrix(graph, n0, n1)
        del graph
        solution = solve_ilp(crossing_matrix, n1, barycenter_crossing)
        order = get_order(solution, n0, n1)
        del solution
    else:
        fil_graph, updated_n0, updated_n1, new_final_to_original, zero_deg_node_b = preprocess1(graph, deg, n0, n1)

        barycenter_crossing = get_barycenter_crossings(fil_graph, updated_n0, updated_n1)
        crossing_matrix = get_crossing_matrix(fil_graph, updated_n0, updated_n1)
        del fil_graph
        solution = solve_ilp(crossing_matrix, updated_n1, barycenter_crossing)
        order = get_order(solution, updated_n0, updated_n1)
        del solution
        order = [new_final_to_original[i] for i in order]
        for i in zero_deg_node_b:
            order.append(i)
    
    for element in order:
        print(element)

if __name__ == '__main__':
    # start_time = time.perf_counter()
    main()
    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # print('Elapsed time:', elapsed_time)