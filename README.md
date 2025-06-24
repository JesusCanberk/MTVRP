# MTVRP: Multi-Vehicle Routing Problem Solver with Time Windows

## Overview

**MTVRP** is a Python-based application for solving the Multi-Vehicle Routing Problem with Time Windows (MVRPTW) using advanced metaheuristic algorithms. The program features a user-friendly GUI that allows for loading standard VRPTW instances, configuring algorithm parameters, running solutions, and interactively visualizing the results. All solutions include detailed feasibility checks and can be saved for later analysis.

---

## Algorithm Details

- **Genetic Algorithm (GA)**: Utilizes tournament selection, ordered crossover, and mutation operations (2-opt and node-move). Infeasible routes and unserved customers are penalized in the cost function.
- **Simulated Annealing (SA)**: Applies local search operators (2-opt, node move) with a cooling schedule to escape local minima and minimize the sum of route distance and penalty costs.
- **Tabu Search**: Iteratively refines routes using memory structures (tabu list and tenure), candidate exploration, and feasibility filtering to find cost-effective solutions.

All algorithms consider:
- Vehicle capacity constraints
- Time window constraints for service at each customer
- Penalties for any constraint violations

---

## Main Steps

1. **Select Instance**: Load a Solomon/Cordeau-style `.txt` file specifying customers, demands, coordinates, and time windows.
2. **Select Algorithm**: Choose between Genetic Algorithm, Simulated Annealing, or Tabu Search.
3. **Set Parameters**: Adjust the relevant algorithm parameters within the GUI.
4. **Solve**: Run the selected algorithm; progress, feasibility, and solution summaries will be shown.
5. **Visualize**: Select vehicle routes to plot and review their details interactively.
6. **Save/Load Solutions**: Solution files are automatically saved with route details and can be revisited later.

---
