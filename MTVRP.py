# main.py
import pandas as pd
import random
import math
import copy
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import matplotlib.pyplot as plt
import os
import datetime


def dist(a, b):
    return ((a.iloc[0] - b.iloc[0]) ** 2 + (a.iloc[1] - b.iloc[1]) ** 2) ** 0.5

def compute_used_capacity(route, demand):
    return sum(demand[node] for node in route if node != 0)

def load_dataset(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    capacity_line = [line for line in lines if "CAPACITY" in line]
    capacity = int(lines[lines.index(capacity_line[0]) + 1].split()[-1])
    start_idx = next(i for i, line in enumerate(lines) if "CUSTOMER" in line)
    raw_data = [line.strip().split() for line in lines[start_idx + 2:] if line.strip()]
    columns = ['CUST NO.', 'XCOORD.', 'YCOORD.', 'DEMAND', 'READY TIME', 'DUE DATE', 'SERVICE TIME']
    df = pd.DataFrame(raw_data, columns=columns).astype(float).astype({'CUST NO.': int})
    df.set_index('CUST NO.', inplace=True)
    return df, capacity

def route_finish_time(route, coords, ready_time, service_time):
    time_val = 0
    for i in range(1, len(route)):
        prev = route[i - 1]
        curr = route[i]
        travel = dist(coords.loc[prev], coords.loc[curr])
        time_val += travel
        if time_val < ready_time[curr]:
            time_val = ready_time[curr]
        time_val += service_time[curr]
    return time_val

def compute_cost_with_penalty(routes, coords, ready_time, due_date, service_time, penalty_per_time, demand, capacity):
    total_distance = 0
    total_penalty = 0
    for route in routes:
        time_val = 0
        current_capacity = 0
        for i in range(1, len(route)):
            prev = route[i - 1]
            curr = route[i]
            travel_time = dist(coords.loc[prev], coords.loc[curr])
            time_val += travel_time
            if time_val < ready_time[curr]:
                time_val = ready_time[curr]
            if time_val > due_date[curr]:
                total_penalty += (time_val - due_date[curr]) * penalty_per_time
            time_val += service_time[curr]
            current_capacity += demand[curr]
            if current_capacity > capacity:
                total_penalty += (current_capacity - capacity) * penalty_per_time
            total_distance += travel_time
    return total_distance + total_penalty

def close_routes_with_depot(routes):
    closed_routes = []
    for route in routes:
        r = route[:]
        if r[0] != 0:
            r.insert(0, 0)
        if r[-1] != 0:
            r.append(0)
        closed_routes.append(r)
    return closed_routes

def two_opt(routes):
    new_routes = copy.deepcopy(routes)
    idx = random.randint(0, len(new_routes) - 1)
    route = new_routes[idx]
    if len(route) > 3:
        i, j = sorted(random.sample(range(1, len(route) - 1), 2))
        route[i:j+1] = reversed(route[i:j+1])
    new_routes[idx] = route
    return new_routes

def move_node_between_routes(routes):
    new_routes = copy.deepcopy(routes)
    from_idx, to_idx = random.sample(range(len(routes)), 2)
    route_from = new_routes[from_idx]
    route_to = new_routes[to_idx]
    if len(route_from) <= 2:
        return new_routes
    node_idx = random.randint(1, len(route_from) - 1)
    node = route_from.pop(node_idx)
    insert_idx = random.randint(1, len(route_to))
    route_to.insert(insert_idx, node)
    return new_routes

def is_feasible(route, coords, ready_time, due_date, service_time, capacity, demand, empty_lists):
    current_time = 0
    current_capacity = 0
    for i in range(1, len(route)):
        prev_node = route[i - 1]
        node = route[i]
        if node not in empty_lists[prev_node]:
            return False
        travel_time = dist(coords.loc[prev_node], coords.loc[node])
        current_time += travel_time
        if current_time < ready_time[node]:
            current_time = ready_time[node]
        if current_time > due_date[node]:
            return False
        current_time += service_time[node]
        current_capacity += demand[node]
        if current_capacity > capacity:
            return False
    return True

def simulated_annealing(initial_routes, coords, ready_time, due_date, service_time,
                        demand, capacity, empty_lists, T=1000, alpha=0.995, max_iter=100000,
                        return_progress=False, penalty_per_time=10, time_limit=60):
    start_time = time.time()
    current = copy.deepcopy(initial_routes)
    best_solution_routes = copy.deepcopy(current)
    best_cost_progress = []
    current_cost = compute_cost_with_penalty(close_routes_with_depot(current), coords, ready_time, due_date, service_time, penalty_per_time, demand, capacity)
    best_solution_cost = current_cost

    for iter_num in range(1, max_iter + 1):
        if time.time() - start_time >= time_limit:
            break

        neighbor = two_opt(current) if random.random() < 0.5 else move_node_between_routes(current)
        all_visited = set()
        for route in neighbor:
            all_visited.update(node for node in route if node != 0)
        if all_visited != set(range(1, len(coords))):
            best_cost_progress.append(best_solution_cost)
            continue
        closed_neighbor = close_routes_with_depot(neighbor)
        neighbor_cost = compute_cost_with_penalty(closed_neighbor, coords, ready_time, due_date, service_time, penalty_per_time, demand, capacity)
        delta = neighbor_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / T):
            current = neighbor
            current_cost = neighbor_cost
            if neighbor_cost < best_solution_cost:
                best_solution_routes = copy.deepcopy(neighbor)
                best_solution_cost = neighbor_cost
        best_cost_progress.append(best_solution_cost)
        T *= alpha

    best_solution_routes = close_routes_with_depot(best_solution_routes)
    final_cost = compute_cost_with_penalty(best_solution_routes, coords, ready_time, due_date, service_time, penalty_per_time, demand, capacity)
    if abs(final_cost - best_solution_cost) > 1e-6:
        best_solution_cost = final_cost
        best_cost_progress[-1] = final_cost
    if return_progress:
        return best_solution_routes, final_cost, best_cost_progress
    else:
        return best_solution_routes, final_cost

def tabu_search(initial_routes, coords, ready_time, due_date, service_time,
                demand, capacity, empty_lists, tenure=20, max_iter=100000,
                return_progress=False, penalty_per_time=10, time_limit=60):
    start_time = time.time()
    best_solution_routes = copy.deepcopy(initial_routes)
    best_cost_progress = []
    current = copy.deepcopy(initial_routes)
    tabu_list = []
    current_cost = compute_cost_with_penalty(close_routes_with_depot(current), coords, ready_time, due_date, service_time, penalty_per_time, demand, capacity)
    best_solution_cost = current_cost

    for iter_num in range(1, max_iter + 1):
        if time.time() - start_time >= time_limit:
            break

        neighborhood = []
        for _ in range(20):
            neighbor = two_opt(current) if random.random() < 0.5 else move_node_between_routes(current)
            if neighbor not in tabu_list:
                all_visited = set()
                for route in neighbor:
                    all_visited.update(node for node in route if node != 0)
                if all_visited != set(range(1, len(coords))):
                    continue
                unreachable = False
                for route in neighbor:
                    for i in range(1, len(route)):
                        if route[i] not in empty_lists[route[i-1]]:
                            unreachable = True
                            break
                    if unreachable:
                        break
                if unreachable:
                    continue
                closed_neighbor = close_routes_with_depot(neighbor)
                cost = compute_cost_with_penalty(closed_neighbor, coords, ready_time, due_date, service_time, penalty_per_time, demand, capacity)
                neighborhood.append((neighbor, cost))
        if neighborhood:
            neighborhood.sort(key=lambda x: x[1])
            current, current_cost = neighborhood[0]
            if current_cost < best_solution_cost:
                best_solution_routes = copy.deepcopy(current)
                best_solution_cost = current_cost
            tabu_list.append(current)
            if len(tabu_list) > tenure:
                tabu_list.pop(0)
        best_cost_progress.append(best_solution_cost)

    best_solution_routes = close_routes_with_depot(best_solution_routes)
    final_cost = compute_cost_with_penalty(best_solution_routes, coords, ready_time, due_date, service_time, penalty_per_time, demand, capacity)
    if abs(final_cost - best_solution_cost) > 1e-6:
        best_solution_cost = final_cost
        best_cost_progress[-1] = final_cost
    if return_progress:
        return best_solution_routes, final_cost, best_cost_progress
    else:
        return best_solution_routes, final_cost

def genetic_algorithm(initial_routes, coords, ready_time, due_date, service_time,
                      demand, capacity, empty_lists, population_size=50, generations=100000,
                      crossover_rate=0.8, mutation_rate=0.1, tournament_size=3,
                      penalty_unserved=10000, penalty_infeasible=5000,
                      return_progress=False, penalty_per_time=10, time_limit=60):
    start_time = time.time()
    population = []
    best_solution = copy.deepcopy(initial_routes)
    for route in best_solution:
        if route[-1] == 0 and len(route) > 1:
            route.pop()
    population.append(best_solution)
    all_customers = list(range(1, len(coords)))
    for _ in range(population_size - 1):
        random.shuffle(all_customers)
        cut1 = random.randint(0, len(all_customers))
        cut2 = random.randint(0, len(all_customers))
        if cut1 > cut2:
            cut1, cut2 = cut2, cut1
        route1 = [0] + all_customers[:cut1]
        route2 = [0] + all_customers[cut1:cut2]
        route3 = [0] + all_customers[cut2:]
        population.append([route1, route2, route3])
    best_solution_routes = copy.deepcopy(best_solution)
    best_solution_cost = float('inf')
    costs = []
    for indiv in population:
        closed_indiv = close_routes_with_depot(indiv)
        cost = compute_cost_with_penalty(closed_indiv, coords, ready_time, due_date, service_time, penalty_per_time, demand, capacity)
        all_nodes = set(range(1, len(coords)))
        visited = set(node for route in indiv for node in route if node != 0)
        missing_count = len(all_nodes - visited)
        if missing_count > 0:
            cost += missing_count * penalty_unserved
        inf_count = 0
        for route in indiv:
            if not is_feasible(route, coords, ready_time, due_date, service_time, capacity, demand, empty_lists):
                inf_count += 1
        if inf_count > 0:
            cost += inf_count * penalty_infeasible
        costs.append(cost)
        if cost < best_solution_cost:
            best_solution_cost = cost
            best_solution_routes = copy.deepcopy(indiv)
    best_cost_progress = []
    gen = 1
    while True:
        if time.time() - start_time >= time_limit or gen > generations:
            break
        new_population = []
        new_population.append(copy.deepcopy(best_solution_routes))
        while len(new_population) < population_size:
            cand1 = random.sample(range(len(population)), tournament_size)
            best_idx1 = cand1[0]
            for idx in cand1[1:]:
                if costs[idx] < costs[best_idx1]:
                    best_idx1 = idx
            parent1 = population[best_idx1]
            cand2 = random.sample(range(len(population)), tournament_size)
            best_idx2 = cand2[0]
            for idx in cand2[1:]:
                if costs[idx] < costs[best_idx2]:
                    best_idx2 = idx
            parent2 = population[best_idx2]
            if best_idx1 == best_idx2:
                cand2 = random.sample(range(len(population)), tournament_size)
                best_idx2 = cand2[0]
                for idx in cand2[1:]:
                    if costs[idx] < costs[best_idx2]:
                        best_idx2 = idx
                parent2 = population[best_idx2]
            if random.random() < crossover_rate:
                parent1_seq = [node for route in parent1 for node in route if node != 0]
                parent2_seq = [node for route in parent2 for node in route if node != 0]
                n = len(parent1_seq)
                len1_a = len(parent1[0]) - 1
                len2_a = len(parent1[1]) - 1
                len3_a = len(parent1[2]) - 1
                len1_b = len(parent2[0]) - 1
                len2_b = len(parent2[1]) - 1
                len3_b = len(parent2[2]) - 1
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i > j:
                    i, j = j, i
                segment1 = parent1_seq[i:j+1]
                child1_seq = [None] * n
                child1_seq[i:j+1] = segment1
                pos = (j + 1) % n
                for k in range(n):
                    idx = (j + 1 + k) % n
                    if parent2_seq[idx] not in segment1:
                        child1_seq[pos] = parent2_seq[idx]
                        pos = (pos + 1) % n
                segment2 = parent2_seq[i:j+1]
                child2_seq = [None] * n
                child2_seq[i:j+1] = segment2
                pos2 = (j + 1) % n
                for k in range(n):
                    idx = (j + 1 + k) % n
                    if parent1_seq[idx] not in segment2:
                        child2_seq[pos2] = parent1_seq[idx]
                        pos2 = (pos2 + 1) % n
                child1 = [
                    [0] + child1_seq[:len1_a],
                    [0] + child1_seq[len1_a:len1_a + len2_a],
                    [0] + child1_seq[len1_a + len2_a:]
                ]
                child2 = [
                    [0] + child2_seq[:len1_b],
                    [0] + child2_seq[len1_b:len1_b + len2_b],
                    [0] + child2_seq[len1_b + len2_b:]
                ]
            else:
                child1 = copy.deepcopy(parent1)
                child2 = copy.deepcopy(parent2)
            if random.random() < mutation_rate:
                child1 = two_opt(child1) if random.random() < 0.5 else move_node_between_routes(child1)
            if random.random() < mutation_rate:
                child2 = two_opt(child2) if random.random() < 0.5 else move_node_between_routes(child2)
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)
        new_population = new_population[:population_size]
        population = new_population
        costs = []
        for indiv in population:
            closed_indiv = close_routes_with_depot(indiv)
            cost = compute_cost_with_penalty(closed_indiv, coords, ready_time, due_date, service_time, penalty_per_time, demand, capacity)
            all_nodes = set(range(1, len(coords)))
            visited = set(node for route in indiv for node in route if node != 0)
            missing_count = len(all_nodes - visited)
            if missing_count > 0:
                cost += missing_count * penalty_unserved
            inf_count = 0
            for route in indiv:
                if not is_feasible(route, coords, ready_time, due_date, service_time, capacity, demand, empty_lists):
                    inf_count += 1
            if inf_count > 0:
                cost += inf_count * penalty_infeasible
            costs.append(cost)
            if cost < best_solution_cost:
                best_solution_cost = cost
                best_solution_routes = copy.deepcopy(indiv)
        best_cost_progress.append(best_solution_cost)
        gen += 1

    best_solution_routes = close_routes_with_depot(best_solution_routes)
    final_cost = compute_cost_with_penalty(best_solution_routes, coords, ready_time, due_date, service_time, penalty_per_time, demand, capacity)
    if abs(final_cost - best_solution_cost) > 1e-6:
        best_solution_cost = final_cost
        best_cost_progress[-1] = final_cost
    if return_progress:
        return best_solution_routes, final_cost, best_cost_progress
    else:
        return best_solution_routes, final_cost

###########################
# GUI Class
###########################


class VRPGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vehicle Routing Solver")
        self.geometry("1200x850")
        self.bind("q", lambda event: self.destroy())

        self.dataset_path = tk.StringVar()
        self.selected_algo = tk.StringVar()
        self.time_limit = tk.IntVar(value=60)

        # For vehicle selection and plotting
        self.latest_routes = None
        self.latest_customer_coords = None
        self.latest_ready_time = None
        self.latest_due_date = None
        self.latest_service = None
        self.latest_demand = None
        self.latest_capacity = None
        self.latest_title = None

        self.create_widgets()
        self.hide_vehicle_selector()
        self.enable_widgets(step=0)

    def create_widgets(self):
        self.label = ttk.Label(self, text="Vehicle Routing Problem Solver", font=("Arial", 18, "bold"))
        self.label.pack(pady=(18, 6))

        topbar = ttk.Frame(self)
        topbar.pack(fill=tk.X, padx=14, pady=(2, 8))

        self.param_frame = ttk.LabelFrame(topbar, text="Algorithm Parameters")
        self.param_frame.pack(side=tk.LEFT, padx=(0, 14), pady=0, anchor="n")
        self.param_entries = {}
        self.param_defs = {
            "Simulated Annealing": [
                ("T", 1000, float),
                ("alpha", 0.995, float),
                ("max_iter", 100000, int)
            ],
            "Tabu Search": [
                ("tenure", 20, int),
                ("max_iter", 100000, int)
            ],
            "Genetic Algorithm": [
                ("population_size", 100, int),
                ("generations", 100000, int),
                ("crossover_rate", 0.8, float),
                ("mutation_rate", 0.2, float),
                ("tournament_size", 3, int)
            ]
        }

        controls = ttk.Frame(topbar)
        controls.pack(side=tk.LEFT, padx=0, pady=0)
        self.instance_btn = ttk.Button(controls, text="Select Instance", command=self.select_instance, width=18)
        self.instance_btn.pack(side=tk.LEFT, padx=(0,4))

        self.algo_box = ttk.Combobox(
            controls, textvariable=self.selected_algo, state="readonly",
            values=list(self.param_defs.keys()), width=22
        )
        self.algo_box.set("Select Method")
        self.algo_box.pack(side=tk.LEFT, padx=3)
        
        self.time_label = ttk.Label(controls, text="Time (s):")
        self.time_label.pack(side=tk.LEFT, padx=(12,2))
        self.time_entry = ttk.Entry(controls, width=8, textvariable=self.time_limit)
        self.time_entry.pack(side=tk.LEFT, padx=(0,6))
        self.time_entry.bind("<FocusOut>", lambda e: self.validate_time())
        
        self.solve_btn = ttk.Button(controls, text="Solve", command=self.threaded_run, width=10)
        self.solve_btn.pack(side=tk.LEFT, padx=(6,4))

        # New button to show saved solutions
        self.show_btn = ttk.Button(controls, text="Show Solutions", command=self.show_solutions, width=14)
        self.show_btn.pack(side=tk.LEFT)

        self.result_box = tk.Frame(self, bg="#f8f9fb", bd=1, relief="ridge")
        self.result_box.pack(fill=tk.X, padx=24, pady=(4, 2))

        self.result_label = tk.Label(
            self.result_box,
            text="",
            font=("Consolas", 10),
            justify=tk.LEFT,
            anchor="w",
            bg="#f8f9fb",
            padx=5,
            pady=3
        )
        self.result_label.pack(fill=tk.X)
        # ---- Multi-selection Listbox for Vehicle(s) ----
        self.vehicle_box_holder = tk.Frame(self)
        self.vehicle_box_holder.pack(fill=tk.X, padx=16, pady=(4, 0))
        self.vehicle_frame = ttk.LabelFrame(self.vehicle_box_holder, text="Select Vehicles to Display")
        self.vehicle_listbox = tk.Listbox(
            self.vehicle_frame, selectmode="multiple", exportselection=False, height=3, width=16, font=("Arial", 12)
        )
        self.vehicle_listbox.pack()
        for v in ["Vehicle A", "Vehicle B", "Vehicle C"]:
            self.vehicle_listbox.insert(tk.END, v)
        self.vehicle_listbox.bind("<<ListboxSelect>>", self.on_vehicle_selected)

        plots_box = ttk.Frame(self)
        plots_box.pack(fill=tk.BOTH, expand=True, padx=16, pady=(4, 10))
        self.route_frame = ttk.Frame(plots_box)
        self.route_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8), pady=4)
        self.cost_frame = ttk.Frame(plots_box)
        self.cost_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=4)

        self.route_canvas = None
        self.cost_canvas = None


    def show_solutions(self):
        sol_dir = "solutions"
        if not os.path.exists(sol_dir):
            messagebox.showinfo("No Solutions", "No solutions directory found.")
            return
        files = os.listdir(sol_dir)
        if not files:
            messagebox.showinfo("No Solutions", "No solution files found.")
            return
        file_path = filedialog.askopenfilename(
            title="Select Solution File",
            initialdir=sol_dir,
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            top = tk.Toplevel(self)
            top.title(os.path.basename(file_path))
            text = tk.Text(top, wrap="none")
            text.insert("1.0", content)
            text.config(state="disabled")
            text.pack(fill=tk.BOTH, expand=True)
    

    def show_param_fields(self, algo_name):
        for child in self.param_frame.winfo_children():
            child.destroy()
        self.param_entries.clear()
        param_defs = self.param_defs.get(algo_name, [])
        for i, (pname, pdefault, ptype) in enumerate(param_defs):
            label = ttk.Label(self.param_frame, text=f"{pname}:")
            label.grid(row=i, column=0, sticky="e", padx=5, pady=1)
            entry = ttk.Entry(self.param_frame, width=10)
            entry.insert(0, str(pdefault))
            entry.grid(row=i, column=1, sticky="w", padx=5, pady=1)
            self.param_entries[pname] = (entry, ptype)
    def show_vehicle_selector(self):
        self.vehicle_frame.pack(pady=(6, 6))
    def hide_vehicle_selector(self):
        self.vehicle_frame.pack_forget()
    def clear_plots(self):
        if self.route_canvas:
            self.route_canvas.get_tk_widget().destroy()
            self.route_canvas = None
        if self.cost_canvas:
            self.cost_canvas.get_tk_widget().destroy()
            self.cost_canvas = None
        self.hide_vehicle_selector() 

    def print_detailed_route(self, route, coords, ready_time, due_date, service, demand, capacity, vehicle_label):
        print(f"Vehicle {vehicle_label}: {route}")
        time_val = 0
        load = 0
        for i in range(1, len(route)):
            prev = route[i - 1]
            curr = route[i]
            travel = dist(coords.loc[prev], coords.loc[curr])
            time_val += travel
            arrival_time = max(time_val, ready_time[curr])
            status = "on time" if arrival_time <= due_date[curr] else f"late by {arrival_time - due_date[curr]:.2f}"
            print(
                f"  Arrive at {curr} at time {arrival_time:.2f} ({status}), Load = {load + (demand[curr] if curr != 0 else 0)}/{capacity}")
            time_val = arrival_time
            if curr != 0:
                time_val += service[curr]
                print(f"    Departure from {curr} at {time_val:.2f}")
                load += demand[curr]
        print(f"  Route finish time: {time_val:.2f}\n")

    def enable_widgets(self, step=0):
        self.algo_box.config(state="disabled")
        self.time_entry.config(state="disabled")
        self.solve_btn.config(state="disabled")
        if step >= 1:
            self.algo_box.config(state="readonly")
        if step >= 2:
            self.time_entry.config(state="normal")
        if step >= 3:
            self.solve_btn.config(state="normal")

    def check_ready_to_solve(self):
        instance_ok = bool(self.dataset_path.get())
        algo_ok = self.selected_algo.get() in [
            "Simulated Annealing", "Tabu Search", "Genetic Algorithm"
        ]
        try:
            t = int(self.time_limit.get())
            time_ok = 60 <= t <= 300
        except Exception:
            time_ok = False
        if instance_ok and algo_ok and time_ok:
            self.solve_btn.config(state="normal")
        else:
            self.solve_btn.config(state="disabled")

    def select_instance(self):
        file_path = filedialog.askopenfilename(
            title="Select Instance File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not file_path:
            return
        self.dataset_path.set(file_path)

        if self.route_canvas:
            self.route_canvas.get_tk_widget().destroy()
            self.route_canvas = None
        if self.cost_canvas:
            self.cost_canvas.get_tk_widget().destroy()
            self.cost_canvas = None

        self.result_label.config(text=f"Selected: {os.path.basename(file_path)}. Now select algorithm.")
        self.algo_box.bind("<<ComboboxSelected>>", self.on_algo_selected)
        self.enable_widgets(step=1)
        self.check_ready_to_solve()

    def get_and_validate_params(self):
        algo = self.selected_algo.get()
        param_defs = self.param_defs.get(algo, [])
        params = {}
        for pname, pdefault, ptype in param_defs:
            entry, typ = self.param_entries[pname]
            val_str = entry.get()
            try:
                if typ == int:
                    val = int(val_str)
                    if val <= 0:
                        raise ValueError
                elif typ == float:
                    val = float(val_str)
                    if val <= 0:
                        raise ValueError
                    if pname in ["alpha", "crossover_rate", "mutation_rate"]:
                        if not (0 < val < 1):
                            raise ValueError
                else:
                    val = val_str
            except Exception:
                messagebox.showerror(
                    "Invalid Parameter",
                    f"Parameter '{pname}' must be positive{' and between 0 and 1' if pname in ['alpha', 'crossover_rate', 'mutation_rate'] else ''}."
                )
                return None
            params[pname] = val
        return params

    def on_algo_selected(self, event=None):
        self.param_frame.pack_forget()  # Remove from top
        self.param_frame.pack(side=tk.BOTTOM, padx=14, pady=(2, 8), anchor="w")  # Reattach below
        self.clear_plots()
        self.show_param_fields(self.selected_algo.get())
        
        filename = os.path.basename(self.dataset_path.get())
        algo = self.selected_algo.get()
        self.result_label.config(
            text=f"Instance {filename} will be solved using {algo}.",
            anchor="w", justify="left"
        )
        self.enable_widgets(step=2)
        self.check_ready_to_solve()

    def validate_time(self, event=None):
        try:
            t = int(self.time_limit.get())
            if not (60 <= t <= 300):
                raise ValueError
            self.enable_widgets(step=3)
            self.clear_plots()
        except Exception:
            messagebox.showwarning("Invalid Input", "Please enter a time limit between 60 and 300 seconds.")
            self.time_limit.set(60)
            self.enable_widgets(step=2)
        self.check_ready_to_solve()

    def threaded_run(self):
        self.param_frame.pack_forget()
        self.solve_btn.config(state="disabled")
        threading.Thread(target=self.run_algorithm, daemon=True).start()

    def run_algorithm(self):
        dataset_file = self.dataset_path.get()
        df, capacity = load_dataset(dataset_file)
        customer_coords = df[['XCOORD.', 'YCOORD.']]
        demand = df['DEMAND']
        ready_time = df['READY TIME']
        due_date = df['DUE DATE']
        service = df['SERVICE TIME']

        empty_lists = [[] for _ in range(len(df))]
        for i in df.index:
            for j in df.index:
                if j != 0 and i != j:
                    travel = dist(customer_coords.loc[i], customer_coords.loc[j])
                    if travel + ready_time[i] + service[i] <= due_date[j]:
                        empty_lists[i].append(j)

        visited_list = [0] * len(df)
        path_a, path_b, path_c = [0], [0], [0]
        current_time_a = current_time_b = current_time_c = 0
        while sum(visited_list) < len(df) - 1:
            for path, current_time in zip([path_a, path_b, path_c],
                                          [current_time_a, current_time_b, current_time_c]):
                last_node = path[-1]
                best_index = None
                best_arrival = float('inf')
                for i in empty_lists[last_node]:
                    if visited_list[i] == 0 and current_time < due_date[i]:
                        arrival = current_time + dist(customer_coords.loc[last_node], customer_coords.loc[i])
                        if arrival < ready_time[i]:
                            arrival = ready_time[i]
                        if arrival < best_arrival:
                            best_arrival = arrival
                            best_index = i
                if best_index is not None:
                    path.append(best_index)
                    visited_list[best_index] = 1
                    travel = dist(customer_coords.loc[last_node], customer_coords.loc[best_index])
                    current_time += travel
                    if current_time < ready_time[best_index]:
                        current_time = ready_time[best_index]
                    current_time += service[best_index]
        initial_routes = [path_a, path_b, path_c]

        print("\n--- Initial Solution ---")
        total_initial_distance = 0
        penalty_per_time = 10
        for v_idx, route in enumerate(initial_routes):
            time_val = 0
            load = 0
            distance = 0
            penalty = 0
            feasible = True
            self.print_detailed_route(
                route,
                customer_coords,
                ready_time,
                due_date,
                service,
                demand,
                capacity,
                vehicle_label=chr(65 + v_idx)
            )
            for i in range(1, len(route)):
                prev = route[i - 1]
                curr = route[i]
                travel = dist(customer_coords.loc[prev], customer_coords.loc[curr])
                time_val += travel
                distance += travel
                if time_val < ready_time[curr]:
                    time_val = ready_time[curr]
                if time_val > due_date[curr]:
                    penalty += (time_val - due_date[curr]) * penalty_per_time
                    feasible = False
                time_val += service[curr]
                load += demand[curr]
                if load > capacity:
                    penalty += (load - capacity) * penalty_per_time
                    feasible = False
            total_cost = distance + penalty
            print(f"Vehicle {chr(65 + v_idx)} Route: {route}")
            print(f"  Used Capacity: {load}/{capacity}")
            print(f"  Finishing Time: {time_val:.2f}")
            print(f"  Distance: {distance:.2f}")
            print(f"  Penalty: {penalty:.2f}")
            print(f"  Total Cost: {total_cost:.2f}")
            print(f"  Feasible: {'Yes ✅' if feasible else 'No ❌'}")
            total_initial_distance += distance
        print(f"Total Distance of All Vehicles: {total_initial_distance:.2f}\n")

        closed_initial_routes = close_routes_with_depot(initial_routes)
        initial_total_cost = compute_cost_with_penalty(
            closed_initial_routes, customer_coords, ready_time, due_date, service,
            penalty_per_time, demand, capacity
        )

        algo = self.selected_algo.get().strip().lower()
        time_limit = int(self.time_limit.get())
        penalty_per_time = 10

        params = self.get_and_validate_params()
        if params is None:
            self.enable_widgets(step=3)
            return

        if algo.startswith("simulated"):
            best_routes, _, best_cost_progress = simulated_annealing(
                initial_routes, customer_coords, ready_time, due_date, service,
                demand, capacity, empty_lists,
                T=params["T"], alpha=params["alpha"], max_iter=params["max_iter"],
                return_progress=True, penalty_per_time=penalty_per_time, time_limit=time_limit
            )
            title = "Best Routes After Simulated Annealing"
        elif algo.startswith("tabu"):
            best_routes, _, best_cost_progress = tabu_search(
                initial_routes, customer_coords, ready_time, due_date, service,
                demand, capacity, empty_lists,
                tenure=params["tenure"], max_iter=params["max_iter"], return_progress=True,
                penalty_per_time=penalty_per_time, time_limit=time_limit
            )
            title = "Best Routes After Tabu Search"
        elif algo.startswith("genetic"):
            best_routes, _, best_cost_progress = genetic_algorithm(
                initial_routes, customer_coords, ready_time, due_date, service,
                demand, capacity, empty_lists,
                population_size=params["population_size"],
                generations=params["generations"],
                crossover_rate=params["crossover_rate"],
                mutation_rate=params["mutation_rate"],
                tournament_size=params["tournament_size"],
                penalty_unserved=10000,
                penalty_infeasible=5000,
                return_progress=True, penalty_per_time=penalty_per_time, time_limit=time_limit
            )
            title = "Best Routes After Genetic Algorithm"
        if isinstance(best_cost_progress, list):
            if len(best_cost_progress) == 0 or not abs(best_cost_progress[0] - initial_total_cost) < 1e-5:
                best_cost_progress = [initial_total_cost] + best_cost_progress
        else:
            self.enable_widgets(step=3)
            return
        for i in range(1, len(best_cost_progress)):
            if best_cost_progress[i] > best_cost_progress[i - 1]:
                best_cost_progress[i] = best_cost_progress[i - 1]

        print("\n--- Best Solution (After Optimization) ---")
        total_best_distance = 0
        penalty_per_time = 10
        any_infeasible = False
        for v_idx, route in enumerate(best_routes):
            time_val = 0
            load = 0
            distance = 0
            penalty = 0
            feasible = True
            self.print_detailed_route(
                route,
                customer_coords,
                ready_time,
                due_date,
                service,
                demand,
                capacity,
                vehicle_label=chr(65 + v_idx)
            )
            for i in range(1, len(route)):
                prev = route[i - 1]
                curr = route[i]
                travel = dist(customer_coords.loc[prev], customer_coords.loc[curr])
                time_val += travel
                distance += travel
                if time_val < ready_time[curr]:
                    time_val = ready_time[curr]
                if time_val > due_date[curr]:
                    penalty += (time_val - due_date[curr]) * penalty_per_time
                    feasible = False
                time_val += service[curr]
                load += demand[curr]
                if load > capacity:
                    penalty += (load - capacity) * penalty_per_time
                    feasible = False
            total_cost = distance + penalty
            print(f"Vehicle {chr(65 + v_idx)} Route: {route}")
            print(f"  Used Capacity: {load}/{capacity}")
            print(f"  Finishing Time: {time_val:.2f}")
            print(f"  Distance: {distance:.2f}")
            print(f"  Penalty: {penalty:.2f}")
            print(f"  Total Cost: {total_cost:.2f}")
            print(f"  Feasible: {'Yes ✅' if feasible else 'No ❌'}")
            total_best_distance += distance
            if not feasible:
                any_infeasible = True
        print(f"Total Distance of All Vehicles (Best Solution): {total_best_distance:.2f}\n")
        if any_infeasible:
            self.after(0, lambda: messagebox.showwarning(
                "Infeasible Solution",
                "Solution is infeasible. Try to chang running time or parameters."
            ))

        total_dist = sum(
            dist(customer_coords.loc[route[i - 1]], customer_coords.loc[route[i]])
            for route in best_routes for i in range(1, len(route))
        )
        best_cost = best_cost_progress[-1]
        route_summary = f"Best Total Cost ({self.selected_algo.get()}): {best_cost:.2f}\n\n"
        for idx, route in enumerate(best_routes):
            finish_time = route_finish_time(route, customer_coords, ready_time, service)
            used_cap = compute_used_capacity(route, demand)
            penalty = self.compute_route_penalty(route, customer_coords, ready_time, due_date,
                                                service, penalty_per_time, demand, capacity)
            route_distance = 0
            for j in range(1, len(route)):
                route_distance += dist(customer_coords.loc[route[j - 1]], customer_coords.loc[route[j]])
            route_summary += (
                f"Vehicle {chr(65 + idx)} Route: {route}\n"
                f"Used Capacity: {used_cap}/{capacity}\n"
                f"Finishing time: {finish_time:.2f}\n"
                f"Penalty cost: {penalty:.2f}\n"
                f"Route distance: {route_distance:.2f}\n\n"
            )
        for route in best_routes:
            if route[-1] != 0:
                route.append(0)

        self.result_label.config(text=route_summary, anchor='center', justify='center')


        self.latest_routes = best_routes
        self.latest_customer_coords = customer_coords
        self.latest_ready_time = ready_time
        self.latest_due_date = due_date
        self.latest_service = service
        self.latest_demand = demand
        self.latest_capacity = capacity
        self.latest_title = title

        self.after(0, self.draw_route_plot, best_routes, customer_coords, title,
                   ready_time, due_date, service, demand, capacity)
        self.after(0, self.draw_cost_progress_plot, best_cost_progress, algo)
        self.after(0, self.show_vehicle_selector)

        os.makedirs("solutions", exist_ok=True)
        solution_file = os.path.join(
            "solutions",
            f"solution_{os.path.basename(dataset_file)}_{self.selected_algo.get().replace(' ', '')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(solution_file, "w", encoding="utf-8") as f:
            # Header
            f.write(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] ({self.selected_algo.get()}) Instance: {os.path.basename(dataset_file)}, best total cost: {best_cost:.2f}\n")
            f.write(f"Time window used: {int(self.time_limit.get())} seconds\n\n")

            # Initial Solution Detailed
            f.write("--- Initial Solution Detailed ---\n")
            for v_idx, route in enumerate(initial_routes):
                label = chr(65 + v_idx)
                f.write(f"Vehicle {label}: {route}\n")
                time_val = 0.0
                load = 0.0
                for i in range(1, len(route)):
                    prev, curr = route[i-1], route[i]
                    travel = dist(customer_coords.loc[prev], customer_coords.loc[curr])
                    time_val += travel
                    arrival = max(time_val, ready_time[curr])
                    status = "on time" if arrival <= due_date[curr] else f"late by {arrival - due_date[curr]:.2f}"
                    load += demand[curr]
                    f.write(f"  Arrive at {curr} at time {arrival:.2f} ({status}), Load = {load}/{capacity}\n")
                    departure = arrival + (service[curr] if curr != 0 else 0)
                    if curr != 0:
                        f.write(f"    Departure from {curr} at {departure:.2f}\n")
                    time_val = departure
                f.write(f"  Route finish time: {time_val:.2f}\n\n")

            # Initial Solution Summary
            f.write("--- Initial Solution Summary ---\n")
            total_initial_penalty = 0
            total_initial_distance = 0
            for v_idx, route in enumerate(initial_routes):
                label = chr(65 + v_idx)
                used = compute_used_capacity(route, demand)
                finish = route_finish_time(route, customer_coords, ready_time, service)
                dist_sum = sum(dist(customer_coords.loc[route[j-1]], customer_coords.loc[route[j]]) for j in range(1, len(route)))
                penalty = self.compute_route_penalty(route, customer_coords, ready_time, due_date, service, penalty_per_time, demand, capacity)
                total_cost = dist_sum + penalty
                feasible = 'Yes ✅' if penalty == 0 else 'No ❌'
                f.write(f"Vehicle {label} Route: {route}\n")
                f.write(f"  Used Capacity: {used}/{capacity}\n")
                f.write(f"  Finishing Time: {finish:.2f}\n")
                f.write(f"  Distance: {dist_sum:.2f}\n")
                f.write(f"  Penalty: {penalty:.2f}\n")
                f.write(f"  Total Cost: {total_cost:.2f}\n")
                f.write(f"  Feasible: {feasible}\n")
                total_initial_penalty += penalty
                total_initial_distance += dist_sum
            total_initial_cost = total_initial_distance + total_initial_penalty
            #f.write(f"Total Distance Initial: {sum(sum(dist(customer_coords.loc[r[j-1]], customer_coords.loc[r[j]]) for j in range(1,len(r))) for r in initial_routes):.2f}\n\n")
            f.write(f"Total Distance Initial: {total_initial_distance:.2f}\n")
            f.write(f"Total Penalty Cost Initial: {total_initial_penalty:.2f}\n")
            f.write(f"Total Cost Initial (Distance + Penalty): {total_initial_cost:.2f}\n\n")
            # Best Solution Detailed
            f.write("--- Best Solution Detailed ---\n")
            for v_idx, route in enumerate(best_routes):
                label = chr(65 + v_idx)
                f.write(f"Vehicle {label}: {route}\n")
                time_val = 0.0
                load = 0.0
                for i in range(1, len(route)):
                    prev, curr = route[i-1], route[i]
                    travel = dist(customer_coords.loc[prev], customer_coords.loc[curr])
                    time_val += travel
                    arrival = max(time_val, ready_time[curr])
                    status = "on time" if arrival <= due_date[curr] else f"late by {arrival - due_date[curr]:.2f}"
                    load += demand[curr]
                    f.write(f"  Arrive at {curr} at time {arrival:.2f} ({status}), Load = {load}/{capacity}\n")
                    departure = arrival + (service[curr] if curr != 0 else 0)
                    if curr != 0:
                        f.write(f"    Departure from {curr} at {departure:.2f}\n")
                    time_val = departure
                f.write(f"  Route finish time: {time_val:.2f}\n\n")

            # Best Solution Summary
            f.write("--- Best Solution Summary ---\n")
            total_best_penalty = 0
            total_best_distance = 0
            for v_idx, route in enumerate(best_routes):
                label = chr(65 + v_idx)
                used = compute_used_capacity(route, demand)
                finish = route_finish_time(route, customer_coords, ready_time, service)
                dist_sum = sum(dist(customer_coords.loc[route[j-1]], customer_coords.loc[route[j]]) for j in range(1, len(route)))
                penalty = self.compute_route_penalty(route, customer_coords, ready_time, due_date, service, penalty_per_time, demand, capacity)
                total_cost = dist_sum + penalty
                feasible = 'Yes ✅' if penalty == 0 else 'No ❌'
                f.write(f"Vehicle {label} Route: {route}\n")
                f.write(f"  Used Capacity: {used}/{capacity}\n")
                f.write(f"  Finishing Time: {finish:.2f}\n")
                f.write(f"  Distance: {dist_sum:.2f}\n")
                f.write(f"  Penalty: {penalty:.2f}\n")
                f.write(f"  Total Cost: {total_cost:.2f}\n")
                f.write(f"  Feasible: {feasible}\n")
                total_best_penalty += penalty
                total_best_distance += dist_sum
            """total_dist = sum(
                sum(dist(customer_coords.loc[route[j-1]], customer_coords.loc[route[j]]) for j in range(1, len(route)))
                for route in best_routes
            )"""
            total_best_cost = total_best_distance + total_best_penalty
            f.write(f"Total Distance of All Vehicles (Best Solution): {total_best_distance:.2f}\n")
            f.write(f"Total Penalty Cost (Best Solution): {total_best_penalty:.2f}\n")
            f.write(f"Total Cost (Distance + Penalty, Best Solution): {total_best_cost:.2f}\n")
            #f.write(f"Total Distance of All Vehicles (Best Solution): {total_dist:.2f}\n")

        self.enable_widgets(step=3)

    def on_vehicle_selected(self, event=None):
        # Update the plot according to vehicle selection
        if self.latest_routes is not None:
            self.draw_route_plot(
                self.latest_routes,
                self.latest_customer_coords,
                self.latest_title,
                self.latest_ready_time,
                self.latest_due_date,
                self.latest_service,
                self.latest_demand,
                self.latest_capacity
            )


    def draw_route_plot(self, best_routes, customer_coords, title,
                        ready_time, due_date, service, demand, capacity):
        penalty_per_time = 10
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.scatter(customer_coords['XCOORD.'], customer_coords['YCOORD.'], c='blue', label='Customers', zorder=3)
        ax.scatter(customer_coords.loc[0]['XCOORD.'], customer_coords.loc[0]['YCOORD.'],
                c='black', marker='s', s=90, label='Depot', zorder=4)

        # Get selected vehicle indices
        selected = list(self.vehicle_listbox.curselection())
        if selected:
            vehicle_indices = selected
            colors = ['red', 'green', 'orange']
            for idx in vehicle_indices:
                if idx < len(best_routes):
                    route = best_routes[idx]
                    x = customer_coords.loc[route]['XCOORD.']
                    y = customer_coords.loc[route]['YCOORD.']
                    ax.plot(x, y, color=colors[idx % len(colors)], marker='o', label=f"Vehicle {chr(65 + idx)}", zorder=2)
            ax.legend(loc='upper right', fontsize=10)
        # If nothing selected, don't plot any routes, only show depot and customers (no legend needed except for depot/customers)
        else:
            handles, labels = ax.get_legend_handles_labels()
            # Show only depot and customers in legend
            ax.legend(handles[:2], labels[:2], loc='upper right', fontsize=10)

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("X Coordinate", fontsize=11)
        ax.set_ylabel("Y Coordinate", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        fig.tight_layout()
        if self.route_canvas:
            self.route_canvas.get_tk_widget().destroy()
        self.route_canvas = FigureCanvasTkAgg(fig, master=self.route_frame)
        self.route_canvas.draw()
        self.route_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def draw_cost_progress_plot(self, best_cost_progress, algo):
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(range(1, len(best_cost_progress) + 1), best_cost_progress,
                marker='o', markersize=2.2, color='#117a1b', linewidth=2)
        ax.set_title(f"Best Cost Over Iterations ({algo.upper()})", fontsize=13)
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel("Best Cost", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()
        if self.cost_canvas:
            self.cost_canvas.get_tk_widget().destroy()
        self.cost_canvas = FigureCanvasTkAgg(fig, master=self.cost_frame)
        self.cost_canvas.draw()
        self.cost_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def compute_route_penalty(self, route, coords, ready_time, due_date,
                              service_time, penalty_per_time, demand, capacity):
        time = 0
        capacity_used = 0
        penalty = 0
        for i in range(1, len(route)):
            prev = route[i - 1]
            curr = route[i]
            travel_time = dist(coords.loc[prev], coords.loc[curr])
            time += travel_time
            if time < ready_time[curr]:
                time = ready_time[curr]
            if time > due_date[curr]:
                penalty += (time - due_date[curr]) * penalty_per_time
            time += service_time[curr]
            capacity_used += demand[curr]
            if capacity_used > capacity:
                penalty += (capacity_used - capacity) * penalty_per_time
        return penalty

if __name__ == '__main__':
    app = VRPGUI()
    app.mainloop()
