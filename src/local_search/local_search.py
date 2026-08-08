import random
import copy
import time
import os

NUM_MISSIONS = 5
FLEETS = ['Armada A', 'Armada B', 'Armada C']
PLANETS = ['Planet X', 'Planet Y', 'Planet Z']
DAYS = [1, 2, 3]

def generate_initial_state():
    """Membangkitkan list jadwal misi awal secara acak."""
    return [{'fleet': random.choice(FLEETS), 
             'planet': random.choice(PLANETS), 
             'day': random.choice(DAYS)} for _ in range(NUM_MISSIONS)]

def calculate_cost(state):
    """Menghitung total pelanggaran hard constraint."""
    cost = 0
    for i in range(len(state)):
        for j in range(i + 1, len(state)):
            if state[i]['day'] == state[j]['day']:
                if state[i]['fleet'] == state[j]['fleet']:
                    cost += 1
                if state[i]['planet'] == state[j]['planet']:
                    cost += 1
    return cost

def get_neighbors(state):
    """Mengembalikan daftar state tetangga menggunakan mekanisme Shift dan Swap."""
    neighbors = []
    # Shift
    for i in range(len(state)):
        for fleet in FLEETS:
            if fleet != state[i]['fleet']:
                new_state = copy.deepcopy(state)
                new_state[i]['fleet'] = fleet
                neighbors.append(new_state)
    # Swap
    for i in range(len(state)):
        for j in range(i + 1, len(state)):
            new_state = copy.deepcopy(state)
            new_state[i], new_state[j] = new_state[j], new_state[i]
            neighbors.append(new_state)
    return neighbors

def print_visualization(state, cost, iteration, algo_name, action=""):
    """Membersihkan layar dan mencetak state terbaru untuk efek animasi."""
    os.system('cls' if os.name == 'nt' else 'clear')
    print("[ANIMASI PENCARIAN BERJALAN...]")
    print(f"Iterasi: {iteration} | Algoritma: {algo_name} | Cost Saat Ini: {cost}\n")
    print("Jadwal Terkini:")
    
    for idx, mission in enumerate(state):
        print(f"[Misi {idx}] {mission['fleet']} -> {mission['planet']} (Hari {mission['day']})")
    
    if action:
        print(f"\n>> Action: {action}")
    time.sleep(0.15)

def run_hill_climbing(initial_state, variant):
    """Menjalankan loop pencarian varian Hill-Climbing."""
    current_state = initial_state
    current_cost = calculate_cost(current_state)
    iteration = 0
    restarts = 0
    sideways_count = 0
    
    while current_cost > 0:
        print_visualization(current_state, current_cost, iteration, f"{variant} HC", "Mencari neighbor...")
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
        elif variant == "Sideways Move" and best_cost == current_cost and sideways_count < 10:
            current_state = random.choice([n for n in neighbors if calculate_cost(n) == current_cost])
            sideways_count += 1
        elif variant == "Random Restart":
            current_state = generate_initial_state()
            current_cost = calculate_cost(current_state)
            restarts += 1
        else:
            break
            
        iteration += 1

    print_visualization(current_state, current_cost, iteration, f"{variant} HC", "Selesai.")
    
    print("\n>> PENCARIAN SELESAI <<")
    if current_cost == 0:
        reason = f"Cost 0 tercapai pada iterasi ke-{iteration}"
        reason += f" (Setelah {restarts}x Restart)!" if restarts > 0 else "!"
        print(f"Alasan: {reason}\n")
        print("--- JADWAL AKHIR VALID ---")
        for idx, mission in enumerate(current_state):
            print(f"[Misi {idx}] {mission['fleet']} -> {mission['planet']} (Hari {mission['day']})")
        print(">> Seluruh misi berhasil dijadwalkan tanpa konflik!")
    else:
        print("Alasan: Terjebak di Local Optimum.")

def start():
    """Fungsi inisialisasi dan interaksi menu CLI."""
    print("=== INISIALISASI ASTRO-SCHEDULER ===")
    print(f"Mengatur {NUM_MISSIONS} Misi, {len(FLEETS)} Armada, {len(PLANETS)} Planet, {len(DAYS)} Hari...")
    
    initial_state = generate_initial_state()
    initial_cost = calculate_cost(initial_state)
    
    print(">> Initial State ter-generate (acak)!")
    print(f">> Cost Awal: {initial_cost} (Terdapat {initial_cost} konflik jadwal)\n")
    
    print("Pilih Algoritma:")
    print("1. Hill-Climbing")
    print("2. Simulated Annealing")
    print("3. Genetic Algorithm")
    pilihan_algo = input("Masukkan pilihan (1-3): ")
    
    if pilihan_algo == '1':
        print("\nPilih Varian Hill-Climbing:")
        print("a. Basic (Steepest-Ascent)")
        print("b. Sideways Move")
        print("c. Stochastic")
        print("d. Random Restart")
        pilihan_varian = input("Masukkan pilihan (a-d): ")
        
        varian_map = {'a': 'Basic', 'b': 'Sideways Move', 'c': 'Stochastic', 'd': 'Random Restart'}
        varian = varian_map.get(pilihan_varian, 'Basic')
        run_hill_climbing(initial_state, variant=varian)
    else:
        print("Algoritma belum diimplementasikan di PoC ini.")

if __name__ == "__main__":
    start()