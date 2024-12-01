import numpy as np
import time
import matplotlib.pyplot as plt
import os

# Create a directory for saving outputs
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# LU Decomposition
def lu_decomposition_example(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for i in range(n):
        # Upper Triangular
        for k in range(i, n):
            U[i, k] = A[i, k] - sum(L[i, j] * U[j, k] for j in range(i))
        
        # Lower Triangular
        for k in range(i, n):
            if i == k:
                L[i, i] = 1  # Diagonal as 1
            else:
                L[k, i] = (A[k, i] - sum(L[k, j] * U[j, i] for j in range(i))) / U[i, i]
    return L, U

# LUP Decomposition
def lup_decomposition_example(A):
    n = A.shape[0]
    P = np.eye(n)
    L = np.zeros((n, n))
    U = np.copy(A)

    for i in range(n):
        # Pivoting
        max_index = np.argmax(abs(U[i:, i])) + i
        if i != max_index:
            U[[i, max_index], :] = U[[max_index, i], :]
            P[[i, max_index], :] = P[[max_index, i], :]
        
        # Decompose
        for j in range(i+1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, :] -= factor * U[i, :]

    np.fill_diagonal(L, 1)  # Diagonal of L is 1
    return P, L, U

# Solving Linear Systems
def forward_substitution(L, b):
    n = L.shape[0]
    y = np.zeros_like(b)
    for i in range(n):
        y[i] = b[i] - sum(L[i, j] * y[j] for j in range(i))
    return y

def backward_substitution(U, y):
    n = U.shape[0]
    x = np.zeros_like(y)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i+1, n))) / U[i, i]
    return x

def solve_lup_example(A, b):
    P, L, U = lup_decomposition_example(A)
    b = np.dot(P, b)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x

# Benchmarking
def benchmark_solver_example():
    sizes = [10, 50, 100, 200, 500]
    lup_times = []
    gaussian_times = []

    for n in sizes:
        A = np.random.rand(n, n)
        b = np.random.rand(n)

        # Benchmark LUP decomposition
        start = time.time()
        solve_lup_example(A, b)
        lup_times.append(time.time() - start)

        # Benchmark Gaussian elimination
        start = time.time()
        np.linalg.solve(A, b)
        gaussian_times.append(time.time() - start)

    # Save results to a text file
    benchmark_path = os.path.join(output_dir, "benchmark_results.txt")
    with open(benchmark_path, "w") as f:
        f.write("Matrix Size | LUP Time (s) | Gaussian Elimination Time (s)\n")
        f.write("-" * 50 + "\n")
        for size, lup_time, gaussian_time in zip(sizes, lup_times, gaussian_times):
            f.write(f"{size:10} | {lup_time:14.6f} | {gaussian_time:25.6f}\n")
    print(f"Benchmark results saved to '{benchmark_path}'")

    # Generate a visualized benchmark graph
    plt.figure()
    plt.plot(sizes, lup_times, label="LUP Decomposition", marker="o")
    plt.plot(sizes, gaussian_times, label="Gaussian Elimination", marker="s")
    plt.xlabel("Matrix Size (n)")
    plt.ylabel("Time (s)")
    plt.title("Benchmark: LUP vs Gaussian Elimination")
    plt.legend()
    plt.grid(True)
    benchmark_plot_path = os.path.join(output_dir, "benchmark_plot.png")
    plt.savefig(benchmark_plot_path)
    print(f"Benchmark plot saved as '{benchmark_plot_path}'")

# Main Function to Run All
if __name__ == "__main__":
    # Example Matrices
    A_example = np.array([[1, 2, 0],
                          [3, 4, 4],
                          [5, 6, 3]], dtype=float)
    b_example = np.array([3, 7, 8], dtype=float)
    
    # Run LU Decomposition
    L, U = lu_decomposition_example(A_example)
    print("LU Decomposition")
    print("L:\n", L)
    print("U:\n", U)

    # Run LUP Decomposition
    P, L, U = lup_decomposition_example(A_example)
    print("\nLUP Decomposition")
    print("P:\n", P)
    print("L:\n", L)
    print("U:\n", U)

    # Solve System
    x_solution = solve_lup_example(A_example, b_example)
    print("\nSolution to Ax = b")
    print("x:\n", x_solution)

    # Benchmarking
    print("\nRunning Benchmark...")
    benchmark_solver_example()
