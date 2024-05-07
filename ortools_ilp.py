# I tried using numpy arrays, but it still gives MLE error.

from ortools.linear_solver import pywraplp
import sys
# import time


def get_order(solution, n1, n0):
    column_sums = [sum(col) for col in zip(*solution)]
    column_sums_with_indices = [(i + 1, column_sums[i]) for i in range(n1)]
    sorted_columns = sorted(column_sums_with_indices, key=lambda x: x[1], reverse=False)
    order =  [col[0] + n0 for col in sorted_columns]
    return order


def solve_ilp(crossing_matrix, n1, n0):
    solver = pywraplp.Solver.CreateSolver('CBC')

    if not solver:
        raise Exception("Could not create solver")

    # Define the range of rows and columns
    row = list(range(1, n1 + 1))
    col = list(range(1, n1 + 1))

    # Define the binary decision variables
    m = {}
    for i in row:
        for j in col:
            m[(i, j)] = solver.IntVar(0, 1, f'm_{i}_{j}')

    # Define the objective function to minimize
    solver.Minimize(solver.Sum(m[(i, j)] * crossing_matrix[i - 1][j - 1] for i in row for j in col))

    # Add constraints for transitivity
    for i in row:
        for j in row:
            for k in row:
                if i != j and j != k and k != i:
                    solver.Add(m[(i, j)] + m[(j, k)] - m[(i, k)] <= 1)

    # Add mutual exclusion constraints
    for i in row:
        for j in row:
            if i != j:
                solver.Add(m[(i, j)] + m[(j, i)] == 1)

    # Solve the ILP problem
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        objective_value = solver.Objective().Value()

        # Retrieve the solution
        solution = [[int(m[(i, j)].SolutionValue()) for j in col] for i in row]

        return objective_value, solution
    else:
        raise Exception("No optimal solution found.")


def get_crossing_matrix(graph, n0, n1):
    c = [[0 for _ in range(n1)] for _ in range(n1)]

    for i in range(n1):
        for j in range(n1):
            Nvi = graph[i + n0 + 1] 
            Nvj = graph[j + n0 + 1]  

            crossing = 0
            l = 0  
            r = 0  

            while l < len(Nvi):
                while r < len(Nvj) and Nvi[l] > Nvj[r]:
                    r += 1
                crossing += r
                l += 1

            c[i][j] = crossing
    
    return c


def read_graph():
    p_line = None
    adjacency_list = {}
    complete = False

    for line in sys.stdin:
        if line.startswith('c'):
            continue
        
        if p_line is None:
            if line.startswith('p'):
                p_line = line.strip()
                _, ocr, n0, n1, m = p_line.split()
                n0 = int(n0)
                n1 = int(n1)
                m = int(m)

                if n0 * n1 == m:
                    complete = True
                    return adjacency_list, ocr, n0, n1, m, complete
                
                adjacency_list = {i: [] for i in range(1, n0 + n1 + 1)}
                edge_count = 0
                continue
        
        if edge_count < m:
            u, v = map(int, line.strip().split())
            adjacency_list[u].append(v)
            adjacency_list[v].append(u)
            edge_count += 1
            if edge_count == m:
                break
    
    if edge_count != m:
        raise ValueError("Edge count does not match the specified number of edges.")
    
    return adjacency_list, ocr, n0, n1, m, complete


def main():
    graph, ocr, n0, n1, m, complete = read_graph()

    if complete == True:
        order = list(range(n0 + 1, n0 + n1 + 1))
        for element in order:
            print(element)
        return

    for node in graph:
        graph[node].sort() 
    crossing_matrix = get_crossing_matrix(graph, n0, n1)

    del graph
    objective_value, solution = solve_ilp(crossing_matrix, n1, n0)
    # print('Objective value:', objective_value)
    # print('Order Matrix : ', solution)

    order = get_order(solution, n1, n0)
    for element in order:
        print(element)

    return


if __name__ == '__main__':
    # start_time = time.perf_counter()
    main()
    # end_time = time.perf_counter()  
    # elapsed_time = end_time - start_time 
    # print('elapsed_time', elapsed_time)