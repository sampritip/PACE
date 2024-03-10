from typing import TypeVar, TextIO, Tuple, List, Dict
import pathlib
import segment_tree as sg

PG = TypeVar('PG', bound='PaceGraph')

class PaceGraph:
    # We assume only the right side is flexible
    right: list
    right_order: Dict[int, int]
    left: int

    edgeset: List[Tuple[int, int]]
    edges: Dict[int, List[int]]

    # Build a graph with sides A and B of size a and b. The side B is assumed to be flexible and A to be fixed.
    # If order is given we set the right side to be ordered that way. 
    def __init__(self, a: int, b: int, edges: list = None, order: list = None):
        if order:
            oelements = set(order)
            relements = set(range(a + 1, a + b + 1))
            diff = oelements - relements
            if len(diff) != 0 :
                raise ValueError(f"Error! 'order' contains elements not present in the graph {diff}.")
            else:
                self.right = order
        else:
            self.right = list(range(a + 1, a + b + 1))
        
        self.right_order = {u: p for p, u in enumerate(self.right)}
        self.left = a

        self.edges = {}
        self.edgeset = []
        if edges:
            edges = sorted(edges) # sort all edges such that we get sorted adjacency lists
            for e in edges:
                u = e[0] if e[0] < e[1] else e[1]
                v = e[0] if e[1] < e[0] else e[1]

                if not (u in self.edges.keys()):
                    self.edges[u] = []

                if not (v in self.edges.keys()):
                    self.edges[v] = []

                self.edges[u].append(v)
                self.edges[v].append(u)
                self.edgeset.append((u,v))

    # Return the neighbors of the given vertex
    def neighbors(self, u: int) -> int:
        return self.edges[u]
    
    # Set the order to the given list
    def set_order(self, order: List[int], check: bool = False):
        if check:
            oelements = set(order)
            relements = set(range(len(self.right) + 1, len(self.right) + self.left + 1))
            diff = oelements - relements
            if len(diff) != 0 :
                raise ValueError(f"Error! 'order' contains elements not present in the graph {diff}.")
            else:
                self.right = order

        self.right = order
        self.right_order = {u: p for p, u in enumerate(self.right)}

    # Check if edges (a,b) and (c,d) cross
    def cross(self, a: int, b: int, c: int, d: int) -> bool:
        if a > b:
            a, b = b, a
            
        if c > d:
            c, d = d, c

        b = self.right_order[b]
        d = self.right_order[d]
        
        return (a < c and b > d) or (c < a and d > b)
    
    # Swap the position of two vertices in the order
    def swap(self, a: int, b: int):
        apos = self.right_order[a]
        bpos = self.right_order[b]

        self.right[apos], self.right[bpos] = self.right[bpos], self.right[apos]
        self.right_order[a] = bpos
        self.right_order[b] = apos

    # Count the crossings in the trivial way by iterating all edges and checking if they cross.
    def countcrossings_trivial(self) -> int:
        crossings = 0
        for a in range(1, self.left+1):
            for c in range(a + 1, self.left+1):
                for b in self.neighbors(a):
                    for d in self.neighbors(c):
                        if self.cross(a, b, c, d):
                            crossings += 1

        return int(crossings)
    
    # Count the crossings using a stack-like implementation.
    def countcrossings_stacklike(self) -> int:
        leftside = sorted(self.edgeset, key=lambda e: (e[0], self.right_order[e[1]]), reverse=True)
        rightside = sorted(self.edgeset, key=lambda e: (self.right_order[e[1]], e[0]), reverse=True)
        leftpos = {e: p for p, e in enumerate(leftside)}
        
        openedges = set()
        doneedges = set()
        crossings = 0
        for i in range(0, len(rightside)):
            e = rightside[i]
            
            if e in openedges:
                openedges.remove(e)
                
            for f in leftside[i:leftpos[e]]:
                if f not in doneedges:
                    openedges.add(f)
            
            ecr = 0
            for f in openedges:
                if self.cross(*e, *f):
                    ecr += 1
                    
            crossings += ecr
            doneedges.add(e)
        
        return crossings
    
    def countcrossings_segtree(self) -> int:
        size = len(self.right)
        edges = sorted(self.edgeset, key=lambda e: (e[0], self.right_order[e[1]]))

        arr = [0] * (size + 1)
        t = sg.SegmentTree(arr)

        crossings = 0
        for edge in edges:
            old = t.query(self.right_order[edge[1]], self.right_order[edge[1]], "sum")
            t.update(self.right_order[edge[1]], old+1)

            if size > self.right_order[edge[1]]+1:
                crossings_found = t.query(self.right_order[edge[1]]+1, size, "sum")
                crossings += crossings_found

        return crossings
            

    # Create a graph from a io object containing gr-formated graph
    @classmethod
    def from_gr(cls, gr: TextIO, order: list = None) -> PG:
        a = 0
        b = 0
        pfound = False
        edges = []
        for line in gr:
            if line[0] == "p":
                a, b = list(map(int, line.split(" ")[2:4]))
                pfound = True
            elif line[0] == "c":
                pass
            elif pfound:
                edges.append(tuple(map(int, line.split(" "))))
            else:
                raise ValueError("ERROR: Encountered edge before p-line.")
                
        return PaceGraph(a, b, edges, order)

    # Return the graph in gr format
    def to_gr(self) -> str:
        newline = "\n"
        thestring = f"p ocr {self.left} {len(self.right)} {len(self.edgeset)}\n"
        thestring = thestring + newline.join([f"{e[0]} {e[1]}" for e in self.edgeset])
    
        return thestring
            
def write_graph(graph: PaceGraph, filename: str):
    with open(filename, "w") as f:
        f.write(graph.to_gr(graph))
        
def write_solution(order: list, filename: str):
    with open(filename, "w") as f:
        for n in order:
            f.write(f"{n}\n")

def read_graph(filename: pathlib.Path, order: list = None) -> PaceGraph:
    with open(filename, "r") as f:
        return PaceGraph.from_gr(f, order)

def read_solution(filename: str) -> list:
    order = []
    with open(filename, "r") as f:
        for line in f.readlines():
            order.append(int(line.strip()))
            
    return order