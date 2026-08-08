import copy
from .environment import FLEETS

def get_neighbors(state):
    neighbors = []
    
    for i in range(len(state)):
        for fleet in FLEETS:
            if fleet != state[i]['fleet']:
                new_state = copy.deepcopy(state)
                new_state[i]['fleet'] = fleet
                neighbors.append(new_state)
                
    for i in range(len(state)):
        for j in range(i + 1, len(state)):
            new_state = copy.deepcopy(state)
            new_state[i], new_state[j] = new_state[j], new_state[i]
            neighbors.append(new_state)
            
    return neighbors