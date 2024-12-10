# M-Height Generator For Analog ECC


## **Overview**

This project provides an efficient solution to the problem of finding optimal generator matrices \( G \) and corresponding codewords \( X \) to minimize the m-height of an analog code. The implementation combines techniques from genetic programming and stochastic optimization to iteratively refine both \( G \)-matrices and \( X \)-vectors for improved performance.

The approach integrates concepts from two tasks:
1. **Task 3**: Optimize \( G \)-matrices to minimize their associated m-heights.
2. **Task 4**: Optimize \( X \)-vectors for a given \( G \)-matrix to maximize the corresponding m-heights.

The solution leverages GPU acceleration with PyTorch for efficient tensor operations and parallelization.

---

## **Problem Statement**

Given:
- \( k \): Number of rows in the generator matrix \( G \).
- \( n \): Number of columns in \( G \).
- \( m \): The height rank to calculate the m-height.

**Objective**:
1. Find \( G \)-matrices that minimize the maximum m-height of their associated codewords.
2. Simultaneously optimize \( X \)-vectors to maximize the m-height for a given \( G \).

**Constraints**:
- \( G \)-matrices must be full rank.
- Elements of \( G \) must be integers.
- The m-height calculation involves selecting the largest and \( m \)-th largest absolute values from codewords \( c = xG \).

---

## **Proposed Solution**

The solution alternates between optimizing \( G \)-matrices and \( X \)-vectors using a hybrid approach that combines **genetic programming** and **stochastic optimization**.

### **Step 1: Initialize**
1. Generate a population of random full-rank \( G \)-matrices.
2. Create a pool of random \( X \)-vectors.

### **Step 2: Iterative Optimization**
For each iteration:
1. **Optimize \( G \)-Matrices**:
   - Evaluate the m-heights for the current population of \( G \)-matrices using the current \( X \)-vectors.
   - Select the top-performing \( G \)-matrices based on m-heights.
   - Apply genetic programming (mutation, crossover, and random generation) to expand the search space for \( G \).

2. **Optimize \( X \)-Vectors**:
   - For each selected \( G \)-matrix, refine its associated \( X \)-vectors to maximize m-heights.
   - Update the pool of \( X \)-vectors with these refined vectors.

### **Step 3: Repeat**
Repeat the above process for a fixed number of iterations to iteratively improve both \( G \) and \( X \).

---

## **Implementation Details**

### **Key Components**
1. **Full-Rank Matrix Generation**:
   Ensures that each generated \( G \)-matrix is full rank by checking its rank condition during initialization.

2. **M-Height Calculation**:
   Efficiently computes the m-height for a given \( G \) and set of \( X \)-vectors by:
   - Multiplying \( X \)-vectors with \( G \).
   - Extracting the largest and \( m \)-th largest absolute values for each codeword.
   - Computing the ratio to determine the m-height.

3. **Genetic Programming**:
   - **Mutation**: Applies small integer perturbations to elements of \( G \) or \( X \).
   - **Crossover**: Combines elements from two matrices or vectors using a binary mask and adds perturbations.
   - **Random Generation**: Introduces random \( G \)-matrices and \( X \)-vectors to maintain diversity.

4. **Parallelization**:
   All operations leverage GPU acceleration for parallel computation using PyTorch tensors.

---

## **Code Usage**

### **Dependencies**
- Python 3.8+
- PyTorch 1.9+ (with CUDA support for GPU acceleration)
- numpy 2.3+
- matplotlib 3.9+
- pickle (for exporting file)
- tqdm (for progress bars)

Install dependencies:
```bash
pip install torch tqdm
```

Clone the Repository:
```bash
git clone https://github.com/username/optimal-generator-matrix.git
cd optimal-generator-matrix
```

Execute the Python file
```bash
python optimize_gs.py
```


