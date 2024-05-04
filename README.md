# TO USE GLPK ILP Solver 

`glpk_script.cpp` utilizes the GNU Linear Programming Kit (GLPK) to create model file 'glpk_model.mod'.

## Prerequisites

- **GLPK**: The GNU Linear Programming Kit (GLPK) is required to solve ILP problems. You can download and install GLPK from [here](https://www.gnu.org/software/glpk/). 

## Usage

1. **Compile `glpk_script.cpp`**:
   - Compile the C++ script using your preferred C++ compiler.
     ```
     g++ glpk_script.cpp 
     ```

2. **Create Input File**:
   - Create a file named `input.gr` in the same directory as `glpk_script.cpp`. This file should contain the input graph instance.

3. **Run `glpk_script`**:
   - Execute the compiled binary. This will generate a file named `glpk_model.mod`.
     ```
     ./a.out
     ```

4. **Solve ILP Problem**:
   - Use `glpsol` command-line utility to solve the ILP problem defined in `glpk_model.mod`. Order Matrix is printed in the `order_matrix.txt` file.
     ```
     time glpsol --model glpk_model.mod
     ```

5. **Get Optimal Permutation**:
   - Compile and run `get_order.cpp` to get the optimal permutation.
     ```
     g++ get_order.cpp -o get_order
     ./get_order
     ```