import numpy as np
import time
import matplotlib.pyplot as plt
import os

# Create a directory for saving outputs
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Check if a matrix is symmetric positive-definite
def is_symmetric(matrix):
    return np.allclose(matrix, matrix.T)

def is_positive_definite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def validate_spd(matrix):
    is_symmetric_check = is_symmetric(matrix)
    is_positive_definite_check = is_positive_definite(matrix)
    print(f"Matrix is symmetric: {is_symmetric_check}")
    print(f"Matrix is positive-definite: {is_positive_definite_check}")
    return is_symmetric_check and is_positive_definite_check

# Least-Squares Fitting Using Cholesky Decomposition
def cholesky_least_squares(X, y):
    # Compute X^T * X
    A = np.dot(X.T, X)
    # Compute X^T * y
    b = np.dot(X.T, y)
    # Solve using Cholesky decomposition
    L = np.linalg.cholesky(A)
    # Forward substitution: Solve L * z = b
    z = np.linalg.solve(L, b)
    # Backward substitution: Solve L^T * x = z
    x = np.linalg.solve(L.T, z)
    return x

# Standard Least-Squares Fitting
def standard_least_squares(X, y):
    # Solve using the normal equation
    A = np.dot(X.T, X)
    b = np.dot(X.T, y)
    return np.linalg.solve(A, b)

# Benchmarking Least-Squares Methods
def benchmark_least_squares():
    sizes = [100, 500, 1000, 2000, 5000]
    cholesky_times = []
    standard_times = []

    for n in sizes:
        X = np.random.rand(n, 2)  # Random design matrix with two columns
        y = np.random.rand(n)

        # Benchmark Cholesky least-squares
        start = time.time()
        cholesky_least_squares(X, y)
        cholesky_times.append(time.time() - start)

        # Benchmark Standard least-squares
        start = time.time()
        standard_least_squares(X, y)
        standard_times.append(time.time() - start)

    # Save benchmark results
    benchmark_path = os.path.join(output_dir, "benchmark_least_squares.txt")
    with open(benchmark_path, "w") as f:
        f.write("Matrix Size | Cholesky Time (s) | Standard Time (s)\n")
        f.write("-" * 50 + "\n")
        for size, chol_time, std_time in zip(sizes, cholesky_times, standard_times):
            f.write(f"{size:10} | {chol_time:14.6f} | {std_time:14.6f}\n")
    print(f"Benchmark results saved to '{benchmark_path}'")

    # Plot benchmark results
    plt.figure()
    plt.plot(sizes, cholesky_times, label="Cholesky Least-Squares", marker="o")
    plt.plot(sizes, standard_times, label="Standard Least-Squares", marker="s")
    plt.xlabel("Matrix Size (n)")
    plt.ylabel("Time (s)")
    plt.title("Benchmark: Cholesky vs Standard Least-Squares")
    plt.legend()
    plt.grid(True)
    benchmark_plot_path = os.path.join(output_dir, "benchmark_least_squares.png")
    plt.savefig(benchmark_plot_path)
    print(f"Benchmark plot saved as '{benchmark_plot_path}'")

# Visualize Least-Squares Fit
def visualize_fit(X, y, coefficients):
    plt.figure()
    plt.scatter(X[:, 1], y, label="Data Points")
    y_fit = np.dot(X, coefficients)
    plt.plot(X[:, 1], y_fit, label="Fitted Line", color="red")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.title("Least-Squares Fit")
    fit_plot_path = os.path.join(output_dir, "least_squares_fit.png")
    plt.savefig(fit_plot_path)
    print(f"Fit visualization saved as '{fit_plot_path}'")

# Main Function
if __name__ == "__main__":
    # Example: Validate Symmetric Positive-Definite Matrix
    A_example = np.array([[4, 1, 2],
                          [1, 3, 0],
                          [2, 0, 5]], dtype=float)
    print("Validating Symmetric Positive-Definite Matrix:")
    validate_spd(A_example)

    # Example Dataset for Least-Squares
    X_example = np.array([[1, 1],
                          [1, 2],
                          [1, 3],
                          [1, 4]])  # Adding a column of ones for the intercept
    y_example = np.array([2, 4, 5, 7])

    # Perform Least-Squares Fit with Cholesky
    print("\nPerforming Least-Squares Fit with Cholesky:")
    coefficients = cholesky_least_squares(X_example, y_example)
    print(f"Coefficients (Intercept, Slope): {coefficients}")

    # Perform Standard Least-Squares Fit
    print("\nPerforming Standard Least-Squares Fit:")
    standard_coefficients = standard_least_squares(X_example, y_example)
    print(f"Coefficients (Intercept, Slope): {standard_coefficients}")

    # Compare and Visualize
    visualize_fit(X_example, y_example, coefficients)

    # Run Benchmarking
    print("\nRunning Benchmark...")
    benchmark_least_squares()
