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
  - [`28.1.py`](28.1/28.1.py): Implementation of LU and LUP decomposition, solving linear equations, and benchmarking against Gaussian elimination.

### **2. Matrix Inversion**
- **Description**: Computes the inverse of a matrix using LUP decomposition and verifies the result.
- **Applications**: Validating the correctness of inversion, comparing performance with NumPy's built-in inversion.
- **Files**:
  - [`28.2.py`](28.2/28.2.py): Implementation of LUP-based matrix inversion, validation, and performance benchmarking.

### **3. Symmetric Positive-Definite Matrices and Least-Squares Approximation**
- **Description**: 
  - Validates symmetric positive-definite (SPD) properties.
  - Implements least-squares approximation using Cholesky decomposition for SPD matrices.
  - Benchmarks runtime against standard least-squares fitting.
- **Files**:
  - [`28.3.py`](28.3/28.3.py): Implementation of SPD matrix validation, least-squares fitting, and runtime comparisons.

## **Benchmark Results**
- **LU and LUP Decomposition**: Comparison of runtime with Gaussian elimination for different matrix sizes. [`benchmark_table_1`](28.1/outputs/benchmark_results.txt)
- **Matrix Inversion**: Performance comparison of LUP-based inversion with NumPy's `linalg.inv`. [`benchmark_table_2`](28.2/outputs/benchmark_inversion_results.txt)
- **Least-Squares Approximation**: Runtime comparison between Cholesky decomposition and standard least-squares fitting for symmetric positive-definite matrices. [`benchmark_table_3`](28.3/outputs/benchmark_least_squares.txt)

### **Visualizations**
- Benchmark plots visualizations are saved in the `outputs/` directory under every sub-chapter folder:
  - [`benchmark_plot_1`](28.1/outputs/benchmark_results.png)
  - [`benchmark_plot_2`](28.2/benchmark_inversion_results.png)
  - [`benchmark_plot_3`](28.3/outputs/benchmark_least_squares.png)


## **References**
1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). Introduction to Algorithms (3rd ed.). MIT Press.
2.  Khan Academy. (n.d.). Forms of linear equations review. Retrieved from https://www.khanacademy.org/math/algebra/x2f8bb11595b61c86:forms-of-linear-equations/x2f8bb11595b61c86:summary-forms-of-two-variable-linear-equations/a/forms-of-linear-equations-review
3. 	GeeksforGeeks. (n.d.). L.U. decomposition of a system of linear equations. Retrieved from https://www.geeksforgeeks.org/l-u-decomposition-system-linear-equations/
4. 	BYJU'S. (n.d.). Inverse of a matrix. Retrieved from https://byjus.com/maths/inverse-matrix/
5. 	GeeksforGeeks. (n.d.). Cholesky decomposition: Matrix decomposition. Retrieved from https://www.geeksforgeeks.org/cholesky-decomposition-matrix-decomposition/
6.	Higham, N. J. (2020, July 21). What is a symmetric positive definite matrix? Retrieved from https://nhigham.com/2020/07/21/what-is-a-symmetric-positive-definite-matrix/

