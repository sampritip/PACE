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
   - Use `glpsol` command-line utility to solve the ILP problem defined in `glpk_model.mod` and output the solution to a file named `solution.txt`.
     ```
     time glpsol --model glpk_model.mod --output solution.txt
     ```

5. **View Solution**:
   - `solution.txt` contains both the minimum number of crossings and the optimal values of the Ordering Matrix, which can be used to derive the optimal order of nodes in the free set. Time taken to solve the model is printed in terminal. 