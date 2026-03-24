import pulp
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def simulate_supply_chain_flow():
    """
    Simulates a Supply Chain Network Flow problem.
    Objective: Minimize transportation costs from Factories to Warehouses 
               while meeting demand and respecting supply limits.
    """
    print("="*50)
    print("1. SUPPLY CHAIN FLOW OPTIMIZATION (Linear Programming)")
    print("="*50)

    # --- Data Definition ---
    factories = ['Factory_A', 'Factory_B']
    warehouses = ['Warehouse_1', 'Warehouse_2', 'Warehouse_3']

    # Max capacity of each factory
    supply = {'Factory_A': 1000, 'Factory_B': 1500}
    
    # Required inventory for each warehouse
    demand = {'Warehouse_1': 800, 'Warehouse_2': 900, 'Warehouse_3': 600}

    # Transportation costs per unit between nodes
    costs = {
        'Factory_A': {'Warehouse_1': 5, 'Warehouse_2': 4, 'Warehouse_3': 3},
        'Factory_B': {'Warehouse_1': 8, 'Warehouse_2': 4, 'Warehouse_3': 3}
    }

    # --- Model Initialization ---
    # We want to minimize the total cost
    model = pulp.LpProblem("Supply_Chain_Minimization", pulp.LpMinimize)

    # --- Decision Variables ---
    # Create a list of all possible routes
    routes = [(f, w) for f in factories for w in warehouses]
    
    # Dictionary containing the decision variables (amount to ship on each route)
    # lowBound=0 ensures non-negativity constraint (cannot ship negative items)
    ship_vars = pulp.LpVariable.dicts("Route", (factories, warehouses), lowBound=0, cat='Integer')

    # --- Objective Function ---
    # Sum of (Units Shipped * Cost per Unit) for all routes
    model += pulp.lpSum([ship_vars[f][w] * costs[f][w] for (f, w) in routes]), "Total_Transportation_Cost"

    # --- Constraints ---
    # 1. Supply Constraints: Factories cannot ship more than they produce
    for f in factories:
        model += pulp.lpSum([ship_vars[f][w] for w in warehouses]) <= supply[f], f"Max_Supply_{f}"

    # 2. Demand Constraints: Warehouses must receive at least what they demand
    for w in warehouses:
        model += pulp.lpSum([ship_vars[f][w] for f in factories]) >= demand[w], f"Min_Demand_{w}"

    # --- Solve and Output ---
    model.solve()

    print(f"Solver Status: {pulp.LpStatus[model.status]}\n")
    
    print("Optimal Shipping Routes:")
    for f in factories:
        for w in warehouses:
            amount = ship_vars[f][w].varValue
            if amount > 0:
                print(f"  - Ship {amount:,.0f} units from {f} to {w} (Cost: ${costs[f][w]}/unit)")
                
    print(f"\nTotal Optimal Cost: ${pulp.value(model.objective):,.2f}\n")


def simulate_vehicle_routing():
    """
    Simulates a Capacitated Vehicle Routing Problem (CVRP).
    Objective: Find the shortest routes for a fleet of vehicles delivering 
               goods from a central depot to various nodes, respecting capacity.
    """
    print("="*50)
    print("2. CAPACITATED VEHICLE ROUTING PROBLEM (Constraint Programming)")
    print("="*50)

    # --- Data Definition ---
    data = {}
    # Distance matrix between locations in meters. Node 0 is the Depot.
    data['distance_matrix'] = [
        [0, 548, 776, 696, 582],     # Node 0 (Depot)
        [548, 0, 684, 308, 194],     # Node 1
        [776, 684, 0, 992, 878],     # Node 2
        [696, 308, 992, 0, 114],     # Node 3
        [582, 194, 878, 114, 0],     # Node 4
    ]
    # Demand units requested at each node (Depot has 0 demand)
    data['demands'] = [0, 1, 1, 2, 4] 
    
    # Fleet setup: 2 vehicles, each can carry a max load of 15
    data['vehicle_capacities'] = [15, 15] 
    data['num_vehicles'] = 2
    data['depot'] = 0

    # --- Model Initialization ---
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']), data['num_vehicles'], data['depot']
    )
    routing = pywrapcp.RoutingModel(manager)

    # --- Callbacks ---
    # 1. Distance Callback: Tells the solver the distance between any two nodes
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    
    # Set the objective: minimize the total distance across all vehicles
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 2. Demand Callback: Tells the solver the demand at each node
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    
    # --- Constraints ---
    # Add Capacity constraint
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack (no extra capacity created on the fly)
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero (vehicles start empty at the depot)
        'Capacity'
    )

    # --- Search Parameters ---
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # Use 'Path Cheapest Arc' as the initial heuristic to find a feasible solution quickly
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # --- Solve and Output ---
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        print_vrrp_solution(data, manager, routing, solution)
    else:
        print("No solution found for the VRP.")


def print_vrrp_solution(data, manager, routing, solution):
    """Utility function to format and print the VRP solution."""
    print(f"Total Fleet Route Distance: {solution.ObjectiveValue()} meters\n")
    
    total_distance = 0
    total_load = 0
    
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = f'-> Route for Delivery Vehicle {vehicle_id}:\n'
        route_distance = 0
        route_load = 0
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += f'  Node {node_index} (Load: {route_load}) -> \n'
            
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            
        node_index = manager.IndexToNode(index)
        plan_output += f'  Node {node_index} (Return to Depot)\n'
        plan_output += f'  -- Route Distance: {route_distance}m\n'
        plan_output += f'  -- Route Load: {route_load} units\n'
        
        print(plan_output)
        total_distance += route_distance
        total_load += route_load


if __name__ == '__main__':
    # Run the Supply Chain Linear Programming model
    simulate_supply_chain_flow()
    
    # Run the Vehicle Routing Constraint Programming model
    simulate_vehicle_routing()