# Quantum Kernel SVM on the Moons Dataset (Qiskit Simulator)

A beginner-friendly implementation of a **quantum kernel method**: a 2-qubit Qiskit circuit is used as a feature map, the kernel (Gram) matrix is computed by simulating state overlaps, and the matrix is passed to a classical scikit-learn SVM. The quantum kernel SVM (QSVM) is benchmarked against standard classical SVM kernels.

Everything runs on an exact **statevector simulator**. No quantum hardware (QPU) and no noise model are used.

## What this project does

1. Generates a 2D, non-linearly separable dataset with `make_moons` (90 samples, noise = 0.15).
2. Splits it into 60 training and 30 test samples.
3. Trains **classical SVM baselines** with linear, polynomial and RBF kernels.
4. Encodes each data point into a 2-qubit state with a custom **quantum feature map**.
5. Computes the **quantum kernel** `K(x, z) = |⟨φ(z)|φ(x)⟩|²` for every pair of points.
6. Trains an SVM with `kernel="precomputed"` on the quantum kernel matrix and evaluates it.

## Method

### Quantum feature map

For an input `x = (x1, x2)`, the circuit on 2 qubits is:

```
H on both qubits          -> superposition
RZ(x1) on q0, RZ(x2) on q1 -> encode features as rotations
CX(0,1) - RZ(x1*x2) on q1 - CX(0,1) -> entanglement and a non-linear cross-term
```

The `x1 * x2` term gives the map a non-linear interaction between features.

### Quantum kernel

The kernel entry is the overlap between two encoded states. The circuit `U(x)` followed by `U†(z)` is simulated with `Statevector`, and the probability of measuring `|00⟩` gives:

```
K(x, z) = |⟨φ(z)|φ(x)⟩|²
```

The Gram matrix is built using symmetry (`K(x, z) = K(z, x)`) to halve the number of circuit evaluations for the training matrix. The notebook checks that the diagonal is all 1s and that the matrix is symmetric.

### Classifier

`sklearn.svm.SVC(kernel="precomputed", C=1.0)` is trained on `K_train` (60 x 60) and evaluated using `K_test` (30 x 60).

## Results

Test accuracy on 30 held-out samples (`random_state=42`):

| Model | Test accuracy |
|---|---|
| Classical SVM, linear kernel | 0.867 |
| Classical SVM, polynomial kernel | 0.900 |
| Classical SVM, RBF kernel | 0.967 |
| **Quantum kernel SVM (2 qubits, simulator)** | **0.800** |

Confusion matrix for the QSVM:

```
[[13  1]
 [ 5 11]]
```

**Takeaway:** on this small, low-dimensional dataset the classical RBF kernel outperforms this quantum kernel. This is expected: the dataset is simple, the feature map is hand-designed and untuned, and there is no reason to expect a quantum advantage on 2D toy data. The value of the project is in understanding and implementing the quantum kernel pipeline end to end, not in beating classical baselines. Results are from a single train/test split on 30 test points, so differences of a few samples are not statistically meaningful.

## Project structure

```
.
├── QML_Quantum_Kernel_Beginner_Simulator.ipynb   # full pipeline
└── README.md
```

## Getting started

```bash
git clone https://github.com/Sadeea2005/Quantum_Kernel_SVM.git
cd Quantum_Kernel_SVM
pip install -r requirements.txt  
jupyter notebook QML_Quantum_Kernel_Beginner_Simulator.ipynb
```

Run the cells from top to bottom. Tested with Qiskit 2.x.

### Requirements

- Python 3.9+
- qiskit, qiskit-aer
- scikit-learn
- numpy
- matplotlib

## Limitations

- Ideal statevector simulation only: no shot noise, no hardware noise.
- The kernel is computed with explicit pairwise simulation (60 x 60 training matrix), which does not scale to large datasets.
- Only one feature map and one dataset were tested; no hyperparameter tuning (e.g. `C`, feature scaling, circuit depth).
- Single random split, so no cross-validation or confidence intervals.

## Possible extensions

- Compare against Qiskit's built-in feature maps (`ZZFeatureMap`) and `qiskit-machine-learning` quantum kernels.
- Add a noise model with `qiskit-aer` and measure how accuracy degrades.
- Add feature scaling, tune `C`, and use cross-validation.
- Try other datasets (e.g. circles, Iris) and more qubits or repeated feature-map layers.
- Run on a real QPU backend, where available.

## Tech stack

Python, Qiskit, scikit-learn, NumPy, Matplotlib, Jupyter

## References

- Havlíček et al., *Supervised learning with quantum-enhanced feature spaces*, Nature 567, 209–212 (2019).
- Qiskit documentation: https://qiskit.org/documentation/
