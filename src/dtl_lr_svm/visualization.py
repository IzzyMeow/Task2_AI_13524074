import matplotlib.pyplot as plt
import numpy as np

def plot_logistic_regression_training(model):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(range(0, model.iterations, 500), model.loss_history, color='b')
    ax1.set_title('Penurunan Loss Function (Learning Curve)')
    ax1.set_xlabel('Iterasi')
    ax1.set_ylabel('Cost / Loss')
    ax1.grid(True)
    
    theta_hist = np.array(model.theta_history)
    
    t1_min, t1_max = theta_hist[:, 0].min() - 0.5, theta_hist[:, 0].max() + 0.5
    t2_min, t2_max = theta_hist[:, 1].min() - 0.5, theta_hist[:, 1].max() + 0.5
    T1, T2 = np.meshgrid(np.linspace(t1_min, t1_max, 50), np.linspace(t2_min, t2_max, 50))
    
    Z = (T1 - theta_hist[-1, 0])**2 + (T2 - theta_hist[-1, 1])**2 
    
    contour = ax2.contour(T1, T2, Z, levels=20, cmap='viridis', alpha=0.5)
    ax2.plot(theta_hist[:, 0], theta_hist[:, 1], marker='o', color='r', markersize=3, label='Lintasan Parameter')
    ax2.plot(theta_hist[-1, 0], theta_hist[-1, 1], marker='x', color='black', markersize=10, label='Titik Optimal (Konvergen)')
    
    ax2.set_title('Lintasan Parameter (Theta 1 vs Theta 2)')
    ax2.set_xlabel('Bobot Fitur 1 (Theta 1)')
    ax2.set_ylabel('Bobot Fitur 2 (Theta 2)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def print_decision_tree(node, depth=0, prefix="Root: "):
    if node is None:
        return
        
    indent = "    " * depth
    if node.value is not None:
        status = "Disetujui" if node.value == 1 else "Ditolak"
        print(f"{indent}{prefix}--> [LEAF] Prediksi: {status}")
    else:
        print(f"{indent}{prefix}[Cabang] Fitur X[{node.feature}] <= {node.threshold:.3f} ?")
        print_decision_tree(node.left, depth + 1, "Ya  (Left) ")
        print_decision_tree(node.right, depth + 1, "Tidak (Right) ")