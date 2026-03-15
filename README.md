# Deep Dynamic Factor Models (DDFM)

This package provides replication codes for the research paper: **"Deep Dynamic Factor Models"**.

**Link to the paper:** [https://arxiv.org/abs/2007.11887](https://arxiv.org/abs/2007.11887)

---

## Architecture Overview

Deep Dynamic Factor Models (DDFM) combine the power of deep learning with the structural interpretability of state-space models. While traditional Dynamic Factor Models (DFMs) rely on linear projections, DDFM utilizes an **Asymmetric Autoencoder** to extract complex, non-linear common factors from high-dimensional datasets.

The project is structured around three main pillars:
1.  **Non-linear Factor Extraction:** Using neural networks (Encoder/Decoder) to handle non-linearities in data.
2.  **State-Space Integration:** Mapping neural network outputs to a Linear Gaussian State-Space (LGSS) model for dynamics and forecasting.
3.  **Robustness to Missing Data:** Specialized algorithms and filters to handle the "ragged edge" and missing values common in macroeconomic data.

---

## Core Components

### 1. `DDFM` ([models/ddfm.py](file:///Users/yannkerzreho/Documents/PR/DDFM/models/ddfm.py))
The central class implementing the model architecture and training logic.
-   **Encoder-Decoder Structure:** The encoder compresses the input data into latent "common factors", while the decoder reconstructs the original observables.
-   **Iterative Training (EM-like):** Implements an iterative algorithm that alternates between:
    -   **Expectation Step:** Using Monte Carlo simulations to handle serial correlation in idiosyncratic components and missing values.
    -   **Maximization Step:** Updating the autoencoder weights using standard backpropagation.
-   **Idiosyncratic Dynamics:** Explicitly models idiosyncratic errors as AR(1) processes to ensure the factors capture only the common variations.

### 2. `StateSpace` ([models/state_space.py](file:///Users/yannkerzreho/Documents/PR/DDFM/models/state_space.py))
Once factors are extracted, this component manages their temporal evolution.
-   **Linear Gaussian State-Space (LGSS):** Maps the weights of the autoencoder and the estimated idiosyncratic variances into a state-space system:
    -   *Measurement:* $z_t = H x_t + v_t$
    -   *Transition:* $x_t = F x_{t-1} + w_t$
-   **Modified Kalman Filter (`KalmanFilterMod`):** An extension of the standard Kalman Filter optimized to handle missing observations using the approach of Shumway and Stoffer (1982).

### 3. `BaiNgModels` ([models/bai_ng_models.py](file:///Users/yannkerzreho/Documents/PR/DDFM/models/bai_ng_models.py))
Provides a suite of benchmark models popularized by Stock & Watson and Bai & Ng.
-   **Principal Component Analysis (PCA):** Standard linear factor estimation.
-   **Targeted Predictors (TPC):** Pre-selection of variables based on their predictive power (t-stats) before applying PCA.
-   **LARS:** Least Angle Regression for high-dimensional feature selection.

---

## Project Structure

```text
DDFM/
├── models/
│   ├── ddfm.py            # Core DDFM implementation
│   ├── state_space.py     # Kalman Filter and LGSS mapping
│   ├── bai_ng_models.py   # Benchmark models (PCA, TPC, LARS)
│   └── base_model.py      # Abstract base class
├── tools/
│   ├── loss_tools.py      # Specialized loss functions for missing data
│   ├── getters_converters_tools.py # State-space parameter extraction
│   └── monthly_quarterly_layer.py  # Mixed frequency handling
├── examples/              # Synthetic and real-world usage examples
└── data/                  # Sample datasets (e.g., FRED-MD snapshots)
```

---

## Installation & Requirements

Ensure you have the following dependencies installed:
-   `tensorflow`
-   `numpy`, `pandas`
-   `pykalman`
-   `scikit-learn`, `statsmodels`

```bash
pip install -r requirements.txt
```
