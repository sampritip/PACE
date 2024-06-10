# filter 0 degree nodes, merge 1 deg nodes, remove 1 deg nodes + 2 ILP

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

def solve_ilp_opt(crossing_matrix, n0, n1, barycenter_crossings, known_order):
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

    # Add constraints for m[(i, i)]
    # for i in range(1, n1 + 1): 
    #     solver.Add(m[(i, j)] == 0)

    # Add constraints for already know order
    for i in range(len(known_order) - 1):
        for j in range(i+1, len(known_order) - 1):
            solver.Add(m[(known_order[i] - n0, known_order[j] - n0)] == 1)
            solver.Add(m[(known_order[j] - n0, known_order[i] - n0)] == 0)

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

def solve_ilp(crossing_matrix, n0, n1, barycenter_crossings):
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

def preprocess3(graph, deg, n0, n1, final_to_original):
    # removing all 1 deg nodes 
    # at this point all children of set A nodes can have only 1 one deg child
    fil_graph = {}
    new_final_to_original = {}
    new_original_to_final = {}

    cur = 0
    for i in range(1, n0+1):
        if deg[i] == 1:
            if deg[graph[i][0]] == 1:
                continue
            else:
                cur += 1
                fil_graph[cur] = graph[i]
                new_final_to_original[cur] = i
                new_original_to_final[i] = cur
        else:
            cur += 1
            new_final_to_original[cur] = i
            new_original_to_final[i] = cur
            fil_graph[cur] = []
            for u in graph[i]:
                if deg[u] == 1:
                    continue
                else:
                    fil_graph[cur].append(u)

    updated_n0 = cur

    for i in range(n0+1, n0+n1+1):
        if deg[i] == 1:
            continue
        else:
            cur += 1
            new_final_to_original[cur] = i
            new_original_to_final[i] = cur
            fil_graph[cur] = []
            for u in graph[i]:
                fil_graph[cur].append(new_original_to_final[u])
    
    for i in range(1, updated_n0+1):
        child_list = []
        for u in fil_graph[i]:
            child_list.append(new_original_to_final[u])
        fil_graph[i] = child_list

    updated_n1 = cur - updated_n0

    return fil_graph, updated_n0, updated_n1, new_final_to_original

def preprocess2(graph, deg, n0, n1, final_to_original):
    # merge sibling nodes in set B with same single parent
    cur = n0
    fil_graph = {}
    new_final_to_original = {}
    #new_original_to_final = {}
    marked = {}
    otf = {}


    for i in range(1, n0+1):
        new_final_to_original[i] = []
        new_final_to_original[i].append(final_to_original[i])
        # new_original_to_final[i] = original_to_final[i]
        # new_original_to_final[final_to_original[i]] = i

        one_deg_nbr = []
        rest_nbr = []
        for u in graph[i]:
            if deg[u] == 1:
                one_deg_nbr.append(u)
            else:
                rest_nbr.append(u)
        
        one_deg_nbr_count = len(one_deg_nbr)
        # rest_nbr_count = len(rest_nbr)
        

        fil_graph[i] = []
        if(one_deg_nbr_count):
            cur += 1
            fil_graph[i].append(cur)
            fil_graph[cur] = []
            fil_graph[cur].append(i)

            new_final_to_original[cur] = []
            for u in one_deg_nbr:  
                marked[u] = 1
                otf[u] = cur
                new_final_to_original[cur].append(final_to_original[u])
         

        for u in rest_nbr:
            if u in marked:
                fil_graph[i].append(otf[u])
                fil_graph[otf[u]].append(i)
            else:
                marked[u] = 1
                cur += 1
                otf[u] = cur

                fil_graph[i].append(cur)
                if(cur not in fil_graph):
                    fil_graph[cur] = []
                fil_graph[cur].append(i)

                if cur not in new_final_to_original:
                    new_final_to_original[cur] = []
                new_final_to_original[cur].append(final_to_original[u])

    updated_n1 = cur - n0

    new_deg = {}
    for i in range(1, n0+updated_n1+1):
        new_deg[i] = len(fil_graph[i])

    return fil_graph, n0, updated_n1, new_final_to_original, new_deg

def preprocess1(graph, deg, n0, n1, min_degree):
    # remove 0 deg nodes from set A and set B
    if min_degree > 0:
        zero_deg_node_b = []
        new_final_to_original = {}
        for i in range(1, n0+n1+1):
            new_final_to_original[i] = i
        return graph, n0, n1, new_final_to_original, deg, zero_deg_node_b

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

    return fil_graph, updated_n0, updated_n1, new_final_to_original, new_deg, zero_deg_node_b

def read_graph():
    adjacency_list = {}
    complete = False
    max_degree = 0
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
                del p_line, ocr
                return adjacency_list, n0, n1, complete, max_degree, deg

            adjacency_list = {i: [] for i in range(1, n0 + n1 + 1)}
            deg = {i: 0 for i in range(1, n0 + n1+ 1)}
            continue

        u, v = map(int, line.strip().split())
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
        deg[u] += 1
        deg[v] += 1
        edge_count += 1
        max_degree = max(max_degree, len(adjacency_list[u]), len(adjacency_list[v]))

    if edge_count != m:
        raise ValueError("Edge count does not match the specified number of edges.")

    return adjacency_list, n0, n1, m, complete, max_degree, deg

def main():
    graph, n0, n1, m, complete, max_degree, deg = read_graph()

    if complete or m == 0:
        order = list(range(n0 + 1, n0 + n1 + 1))
        for element in order:
            print(element)
        return
    
    for node in graph:
        graph[node].sort()
    
    # if max_degree <= 2:
    #     order = median_barycenter(graph, n0, n1)
    #     for element in order:
    #         print(element)
    #     return

    min_degree = min(deg)
    
    # preprocess graph here ---------------------------------------------------------------------------------

    fil_graph, updated_n0, updated_n1, new_final_to_original, new_deg, zero_deg_node_b = preprocess1(graph, deg, n0, n1, min_degree)

    del graph  # Graph is no longer needed after this point

    p2_fil_graph, p2_n0, p2_n1, p2_final_to_original, p2_deg = preprocess2(fil_graph, new_deg, updated_n0, updated_n1, new_final_to_original)

    proceed = False
    for e in deg:
        if e == 1:
            proceed = True

    p1_fil_graph, p1_n0, p1_n1, p1_final_to_original = preprocess3(p2_fil_graph, p2_deg, p2_n0, p2_n1, p2_final_to_original)

    # phase I of ILP solving 
    p1_barycenter_crossing = get_barycenter_crossings(p1_fil_graph, p1_n0, p1_n1)
    p1_crossing_matrix = get_crossing_matrix(p1_fil_graph, p1_n0, p1_n1)
    del p1_fil_graph
    p1_solution = solve_ilp(p1_crossing_matrix, p1_n0, p1_n1, p1_barycenter_crossing)
    del p1_crossing_matrix  # Clear memory once crossing matrix is no longer needed
    p1_order = get_order(p1_solution, p1_n0, p1_n1)
    del p1_solution  # Clear memory once solution is no longer needed

    p1_order = [p1_final_to_original[i] for i in p1_order]

    if proceed == False:
        opt_order = []
        for i in p1_order:
            for u in p2_final_to_original[i]:
                opt_order.append(u)
    else:
        # phase 2 of ILP solving
        p2_barycenter_crossing = get_barycenter_crossings(p2_fil_graph, p2_n0, p2_n1)
        p2_crossing_matrix = get_crossing_matrix(p2_fil_graph, p2_n0, p2_n1)
        del p2_fil_graph
        p2_solution = solve_ilp_opt(p2_crossing_matrix, p2_n0, p2_n1, p2_barycenter_crossing, p1_order)
        del p2_crossing_matrix
        p2_order = get_order(p2_solution, p2_n0, p2_n1)
        del p2_solution

        opt_order = []
        for i in p2_order:
            for u in p2_final_to_original[i]:
                opt_order.append(u)

    for i in zero_deg_node_b:
        opt_order.append(i)

    for element in opt_order:
        print(element)

if __name__ == '__main__':
    # start_time = time.perf_counter()
    main()
    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # print('Elapsed time:', elapsed_time)