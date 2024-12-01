# DAA_Matrix_Operations
# **Matrix Operations: LU Decomposition, Matrix Inversion, and Least-Squares Approximation**

## **Overview**
This repository contains the implementation of key concepts from Chapter 28: *Matrix Operations* of *Introduction to Algorithms, Third Edition* by Thomas H. Cormen et al. The code explores:
- LU and LUP decomposition for solving systems of linear equations.
- Matrix inversion using LUP decomposition.
- Symmetric positive-definite matrices and least-squares approximation using Cholesky decomposition.

## **Contents**
### **1. LU and LUP Decomposition**
- **Description**: Factorizes a square matrix into lower (\( L \)) and upper (\( U \)) triangular matrices, with LUP decomposition adding a permutation matrix (\( P \)) for numerical stability.
- **Applications**: Solving systems of linear equations, inverting matrices, and computing determinants.
- **Files**: 
  - `28.1.py`: Implementation of LU and LUP decomposition, solving linear equations, and benchmarking against Gaussian elimination.

### **2. Matrix Inversion**
- **Description**: Computes the inverse of a matrix using LUP decomposition and verifies the result.
- **Applications**: Validating the correctness of inversion, comparing performance with NumPy's built-in inversion.
- **Files**:
  - `28.2.py`: Implementation of LUP-based matrix inversion, validation, and performance benchmarking.

### **3. Symmetric Positive-Definite Matrices and Least-Squares Approximation**
- **Description**: 
  - Validates symmetric positive-definite (SPD) properties.
  - Implements least-squares approximation using Cholesky decomposition for SPD matrices.
  - Benchmarks runtime against standard least-squares fitting.
- **Files**:
  - `28.3.py`: Implementation of SPD matrix validation, least-squares fitting, and runtime comparisons.

## **Benchmark Results**
- **LU and LUP Decomposition**: Comparison of runtime with Gaussian elimination for different matrix sizes.
- **Matrix Inversion**: Performance comparison of LUP-based inversion with NumPy's `linalg.inv`.
- **Least-Squares Approximation**: Runtime comparison between Cholesky decomposition and standard least-squares fitting for symmetric positive-definite matrices.

### **Visualizations**
- Benchmark plots and least-squares fitting visualizations are saved in the `outputs/` directory:
  - Benchmark runtime plots (e.g., `benchmark_least_squares.png`).
  - Fitted curve visualization (e.g., `least_squares_fit.png`).

## **How to Run**
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
