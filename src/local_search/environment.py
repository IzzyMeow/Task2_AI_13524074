import random

NUM_MISSIONS = 5
FLEETS = ['Armada A', 'Armada B', 'Armada C']
PLANETS = ['Planet X', 'Planet Y', 'Planet Z']
DAYS = [1, 2, 3]

def generate_initial_state():
    return [{'fleet': random.choice(FLEETS), 
             'planet': random.choice(PLANETS), 
             'day': random.choice(DAYS)} for _ in range(NUM_MISSIONS)]

def calculate_cost(state):
    cost = 0
    for i in range(len(state)):
        for j in range(i + 1, len(state)):
            if state[i]['day'] == state[j]['day']:
                if state[i]['fleet'] == state[j]['fleet']:
                    cost += 1
                if state[i]['planet'] == state[j]['planet']:
                    cost += 1
    return cost