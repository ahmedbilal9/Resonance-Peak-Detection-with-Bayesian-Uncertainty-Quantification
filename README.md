# Resonance Peak Detection with Bayesian Uncertainty Quantification

Advanced particle physics analysis for Z boson mass measurement using Bayesian neural networks and Monte Carlo Dropout techniques.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.6+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## Table of Contents

- [Overview](#overview)
- [Technical Background](#technical-background)
  - [Particle Physics Fundamentals](#particle-physics-fundamentals)
  - [Z Boson Mass Measurement](#z-boson-mass-measurement)
  - [Bayesian Neural Networks](#bayesian-neural-networks)
  - [Monte Carlo Dropout](#monte-carlo-dropout)
  - [Uncertainty Quantification](#uncertainty-quantification)
  - [CERN Data Analysis](#cern-data-analysis)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup Instructions](#setup-instructions)
- [Usage](#usage)
  - [Running the Notebook](#running-the-notebook)
  - [Expected Inputs](#expected-inputs)
  - [Expected Outputs](#expected-outputs)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Architecture](#model-architecture)
  - [Training Procedure](#training-procedure)
  - [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Overview

This project explores advanced particle physics analysis with a focus on measuring the Z boson mass using data from CERN experiments. By leveraging Bayesian neural networks and Monte Carlo Dropout techniques, we provide robust uncertainty quantification for high-energy physics measurements.

### Key Features

- **Precision Measurement**: Accurate Z boson mass estimation using statistical analysis
- **Uncertainty Quantification**: Bayesian methods for quantifying measurement uncertainties
- **Monte Carlo Dropout**: Neural network uncertainty estimation through stochastic inference
- **CERN Data Analysis**: Analysis of real particle physics experimental data
- **Resonance Peak Detection**: Advanced algorithms for identifying particle resonances

### Objectives

1. Measure the Z boson mass with high precision
2. Quantify uncertainties in particle physics measurements
3. Validate the Standard Model predictions
4. Demonstrate Bayesian uncertainty quantification in high-energy physics
5. Provide reproducible analysis workflows for particle physics research

---

## Technical Background

### Particle Physics Fundamentals

Particle physics is the branch of physics that studies the nature of particles that constitute matter and radiation. It seeks to understand the fundamental constituents of matter and the forces acting between them. The Standard Model of particle physics describes the electromagnetic, weak, and strong nuclear interactions between elementary particles.

### Z Boson Mass Measurement

The Z boson is a fundamental particle that mediates the weak nuclear force, one of the four fundamental forces of nature. Precise measurement of its mass is critical for:

- Testing the Standard Model of particle physics
- Constraining new physics beyond the Standard Model
- Understanding electroweak symmetry breaking
- Validating quantum field theory predictions

The Z boson mass is approximately 91.2 GeV/c², and its precise measurement requires sophisticated statistical analysis of particle collision data.

### Bayesian Neural Networks

Bayesian neural networks (BNNs) incorporate Bayesian inference into neural network training, treating network weights as probability distributions rather than fixed values. This approach provides:

- **Uncertainty Estimates**: Quantifies both aleatoric and epistemic uncertainties
- **Probabilistic Predictions**: Outputs probability distributions instead of point estimates
- **Robustness**: Better generalization to unseen data
- **Interpretability**: Understanding of model confidence

BNNs are particularly valuable in scientific applications where understanding prediction uncertainty is as important as the prediction itself.

### Monte Carlo Dropout

Monte Carlo Dropout is a practical technique for approximating Bayesian inference in neural networks. The method works by:

1. Applying dropout during both training and inference
2. Making multiple stochastic forward passes through the network
3. Generating a distribution of predictions
4. Computing uncertainty from the prediction variance

This technique provides computationally efficient uncertainty estimation without the full complexity of Bayesian inference.

### Uncertainty Quantification

Uncertainty quantification (UQ) is the systematic study and characterization of uncertainties in computational and real-world systems. In this project, we distinguish between:

- **Aleatoric Uncertainty**: Irreducible uncertainty inherent in the data (measurement noise)
- **Epistemic Uncertainty**: Reducible uncertainty due to limited knowledge (model uncertainty)

Proper uncertainty quantification is essential for:
- Making reliable scientific conclusions
- Understanding measurement limitations
- Comparing theoretical predictions with experimental results

### CERN Data Analysis

CERN (European Organization for Nuclear Research) operates the world's largest particle physics laboratory. The data analyzed in this project comes from particle collision experiments where:

- Protons or other particles collide at high energies
- Detectors record the resulting particle decay products
- The Z boson appears as a resonance peak in the invariant mass distribution
- Statistical analysis extracts particle properties from decay signatures

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or JupyterLab
- At least 4GB of RAM
- Basic understanding of Python and machine learning

### Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/ahmedbilal9/Resonance-Peak-Detection-with-Bayesian-Uncertainty-Quantification.git
cd Resonance-Peak-Detection-with-Bayesian-Uncertainty-Quantification
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**

```bash
jupyter notebook
```

5. **Open the main notebook**

Navigate to `resonance_peak_detection.ipynb` in the Jupyter interface.

---

## Usage

### Running the Notebook

1. Open `resonance_peak_detection.ipynb` in Jupyter Notebook
2. Execute cells sequentially from top to bottom
3. The notebook will load data, train models, and generate visualizations
4. Results will be displayed inline with explanatory text

### Expected Inputs

The notebook expects:
- Particle collision event data (invariant mass measurements)
- Data should contain features relevant to Z boson decay
- CSV or HDF5 format for structured data
- Proper column headers and data types

### Expected Outputs

The analysis produces:
- **Z Boson Mass Estimate**: Central value with confidence intervals
- **Uncertainty Quantification**: Aleatoric and epistemic uncertainty estimates
- **Visualizations**: 
  - Invariant mass distributions
  - Resonance peak plots
  - Posterior probability distributions
  - Uncertainty bands
- **Performance Metrics**: Model accuracy and prediction reliability
- **Statistical Reports**: Detailed analysis results

---

## Project Structure

```
Resonance-Peak-Detection-with-Bayesian-Uncertainty-Quantification/
├── resonance_peak_detection.ipynb    # Main analysis notebook
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── CONTRIBUTING.md                    # Contribution guidelines
└── .gitignore                         # Git ignore rules
```

### File Descriptions

- **resonance_peak_detection.ipynb**: Complete Jupyter notebook containing data loading, preprocessing, model training, evaluation, and visualization
- **README.md**: Comprehensive project documentation
- **requirements.txt**: Python package dependencies with version specifications
- **LICENSE**: MIT License for open-source distribution
- **CONTRIBUTING.md**: Guidelines for contributing to the project
- **.gitignore**: Specifies files and directories to ignore in version control

---

## Methodology

### Data Preprocessing

1. **Data Loading**: Import particle collision event data from CERN experiments
2. **Feature Engineering**: Extract relevant features from raw detector measurements
3. **Data Cleaning**: Remove outliers and handle missing values
4. **Normalization**: Standardize features for neural network training
5. **Train-Test Split**: Divide data into training, validation, and test sets

### Model Architecture

The Bayesian neural network architecture includes:

- **Input Layer**: Accepts particle physics features
- **Hidden Layers**: Dense layers with dropout for uncertainty estimation
- **Activation Functions**: ReLU or similar non-linear activations
- **Dropout Layers**: Applied during both training and inference for Monte Carlo sampling
- **Output Layer**: Produces mass predictions with uncertainty estimates

### Training Procedure

1. **Initialization**: Random weight initialization with proper scaling
2. **Optimization**: Adam or similar gradient-based optimizer
3. **Loss Function**: Mean squared error or negative log-likelihood
4. **Regularization**: Dropout and weight decay to prevent overfitting
5. **Validation**: Monitor performance on held-out validation set
6. **Early Stopping**: Prevent overfitting by stopping when validation performance plateaus

### Evaluation Metrics

- **Mean Absolute Error (MAE)**: Average prediction error magnitude
- **Root Mean Squared Error (RMSE)**: Standard deviation of prediction errors
- **Prediction Interval Coverage**: Fraction of true values within uncertainty bounds
- **Calibration**: Alignment between predicted and actual uncertainties
- **Chi-squared Test**: Statistical comparison with theoretical predictions

---

## Results

The analysis produces several key findings:

### Z Boson Mass Measurement

- Precise estimation of Z boson mass from particle collision data
- Confidence intervals quantifying measurement uncertainty
- Comparison with Standard Model predictions (approximately 91.2 GeV/c²)

### Uncertainty Quantification

- Clear separation of aleatoric and epistemic uncertainties
- Uncertainty estimates for individual predictions
- Model confidence visualization across the mass spectrum

### Visualizations

Key plots include:
- **Invariant Mass Distribution**: Histogram showing Z boson resonance peak
- **Posterior Distribution**: Bayesian posterior for mass parameter
- **Uncertainty Bands**: Prediction intervals showing model confidence
- **Residual Plots**: Analysis of prediction errors
- **Calibration Curves**: Validation of uncertainty estimates

### Performance Metrics

- High prediction accuracy on test data
- Well-calibrated uncertainty estimates
- Strong agreement with theoretical predictions
- Robust performance across different collision energies

---

## Requirements

**Python Version**: 3.8 or higher

**Core Dependencies**:

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >= 1.21.0 | Numerical computing |
| pandas | >= 1.3.0 | Data manipulation |
| matplotlib | >= 3.4.0 | Plotting and visualization |
| scipy | >= 1.7.0 | Scientific computing |
| tensorflow | >= 2.6.0 | Deep learning framework |
| keras | >= 2.6.0 | Neural network API |
| scikit-learn | >= 0.24.0 | Machine learning utilities |
| jupyter | >= 1.0.0 | Interactive notebook environment |

See `requirements.txt` for complete dependency list.

---

## Contributing

We welcome contributions from the community! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:

- Reporting issues
- Submitting pull requests
- Code style guidelines
- Testing procedures

Before contributing, please read our contribution guidelines to ensure a smooth collaboration process.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License is a permissive open-source license that allows for:
- Commercial use
- Modification
- Distribution
- Private use

---

## Acknowledgements

### CERN Collaboration

We gratefully acknowledge the CERN collaboration for providing access to particle physics data and experimental resources. Their pioneering work in high-energy physics has made this research possible.

### Data Sources

- Particle collision event data from CERN experiments
- Standard Model predictions from theoretical physics literature

### References

This project builds upon established research in:

1. **Particle Physics**: Standard Model formulation and Z boson properties
2. **Bayesian Methods**: Uncertainty quantification in scientific computing
3. **Deep Learning**: Neural network architectures for regression tasks
4. **Statistical Analysis**: Maximum likelihood estimation and hypothesis testing

### Key Papers

- Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"
- The ATLAS and CMS Collaborations. "Combined Measurement of the Higgs Boson Mass"
- Bishop, C. M. (2006). "Pattern Recognition and Machine Learning"

---

## Contact

**Author**: Ahmed Bilal

For questions, suggestions, or collaboration opportunities:

- **GitHub**: [ahmedbilal9](https://github.com/ahmedbilal9)
- **Repository**: [Resonance-Peak-Detection-with-Bayesian-Uncertainty-Quantification](https://github.com/ahmedbilal9/Resonance-Peak-Detection-with-Bayesian-Uncertainty-Quantification)

Please open an issue on GitHub for bug reports or feature requests.

---

**Note**: This project is for research and educational purposes. The analysis demonstrates statistical methods for particle physics but should be validated against official CERN publications for production use.