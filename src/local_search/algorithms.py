import random
from .environment import calculate_cost, generate_initial_state
from .neighbors import get_neighbors

def run_hill_climbing(initial_state, variant="Basic", max_sideways=10, max_iterations=1000):
    current_state = initial_state
    current_cost = calculate_cost(current_state)
    
    history = [{'iteration': 0, 'cost': current_cost, 'state': current_state, 'restarts': 0}]
    sideways_count = 0
    restarts = 0
    iteration = 0
    
    while current_cost > 0 and iteration < max_iterations:
        neighbors = get_neighbors(current_state)
        
        if variant == "Stochastic":
            random.shuffle(neighbors)
            best_neighbor = next((n for n in neighbors if calculate_cost(n) < current_cost), None)
            best_cost = calculate_cost(best_neighbor) if best_neighbor else current_cost
        else:
            best_neighbor = min(neighbors, key=calculate_cost)
            best_cost = calculate_cost(best_neighbor)
            
        if best_cost < current_cost:
            current_state = best_neighbor
            current_cost = best_cost
            sideways_count = 0
        elif variant == "Sideways Move" and best_cost == current_cost and sideways_count < max_sideways:
            current_state = random.choice([n for n in neighbors if calculate_cost(n) == current_cost])
            sideways_count += 1
        elif variant == "Random Restart":
            current_state = generate_initial_state()
            current_cost = calculate_cost(current_state)
            restarts += 1
        else:
            break
            
        iteration += 1
        history.append({
            'iteration': iteration, 
            'cost': current_cost, 
            'state': current_state, 
            'restarts': restarts
        })
            
    return current_state, current_cost, history