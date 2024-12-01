import numpy as np
import time
import matplotlib.pyplot as plt
import os

# Create a directory for saving outputs
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# LUP Decomposition
def lup_decomposition(A):
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

# Forward and Backward Substitution
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

# Matrix Inversion using LUP Decomposition
def invert_matrix(A):
    n = A.shape[0]
    P, L, U = lup_decomposition(A)
    A_inv = np.zeros_like(A)
    
    # Solve for each column of the identity matrix
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        b = np.dot(P, e)  # Apply permutation
        y = forward_substitution(L, b)
        A_inv[:, i] = backward_substitution(U, y)
    
    return A_inv

# Matrix Multiplication for Validation
def matrix_multiply(A, B):
    return np.dot(A, B)

# Benchmarking
def benchmark_inversion():
    sizes = [10, 50, 100, 200, 500]
    lup_times = []
    numpy_times = []

    for n in sizes:
        A = np.random.rand(n, n)

        # Benchmark LUP-based inversion
        start = time.time()
        invert_matrix(A)
        lup_times.append(time.time() - start)

        # Benchmark NumPy inversion
        start = time.time()
        np.linalg.inv(A)
        numpy_times.append(time.time() - start)

    # Save benchmark results
    benchmark_path = os.path.join(output_dir, "benchmark_inversion_results.txt")
    with open(benchmark_path, "w") as f:
        f.write("Matrix Size | LUP Inversion Time (s) | NumPy Inversion Time (s)\n")
        f.write("-" * 50 + "\n")
        for size, lup_time, numpy_time in zip(sizes, lup_times, numpy_times):
            f.write(f"{size:10} | {lup_time:21.6f} | {numpy_time:20.6f}\n")
    print(f"Benchmark results saved to '{benchmark_path}'")

    # Plot results
    plt.figure()
    plt.plot(sizes, lup_times, label="LUP Inversion", marker="o")
    plt.plot(sizes, numpy_times, label="NumPy Inversion", marker="s")
    plt.xlabel("Matrix Size (n)")
    plt.ylabel("Time (s)")
    plt.title("Benchmark: LUP vs NumPy Matrix Inversion")
    plt.legend()
    plt.grid(True)
    benchmark_plot_path = os.path.join(output_dir, "benchmark_inversion_plot.png")
    plt.savefig(benchmark_plot_path)
    print(f"Benchmark plot saved as '{benchmark_plot_path}'")

# Main Function
# Main Function
if __name__ == "__main__":
    # Example Matrix
    A_example = np.array([[1, 2, 3],
                          [0, 1, 4],
                          [5, 6, 0]], dtype=float)

    # Compute Inverse
    A_inv = invert_matrix(A_example)
    print("Original Matrix A:\n", A_example)
    print("Inverse of A:\n", A_inv)

    # Verify by multiplying A and its inverse
    I = matrix_multiply(A_example, A_inv)

    # Print the rounded result as integers
    I_rounded = np.rint(I).astype(int)
    print("A * A_inv (Rounded to Integers, Should be Identity):\n", I_rounded)

    # Validate the result
    if np.allclose(I, np.eye(A_example.shape[0])):
        print("Matrix inversion is correct: A * A_inv is the identity matrix.")
    else:
        print("Matrix inversion failed.")

    # Run Benchmarking
    print("\nRunning Benchmark...")
    benchmark_inversion()

