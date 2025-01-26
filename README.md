# Ultra-High-Performance-Concrete-Strength Prediction Using AI Models

This repository contains a Python-based machine learning implementation for predicting the compressive strength of Ultra High Performance Concrete (UHPC) based on various input parameters. The solution uses a variety of machine learning algorithms for regression tasks, optimized using Particle Swarm Optimization (PSO) for hyperparameter tuning.

## Features

- Data preprocessing, including standardization of features.
- Hyperparameter optimization using PSO for multiple regression models.
- Evaluation of models using various performance metrics such as RMSE, MAE, R², and MAPE.
- Cross-validation and performance comparison of multiple models.
- Model optimization and saving the best model for future predictions.
  
![GradientBoosting_cross_validation_plot](https://github.com/user-attachments/assets/33a03cd2-3048-41ae-b6de-baa0234b249f)

## Technologies Used

- Python 3.x
- Libraries:
  - `numpy`, `pandas`, `matplotlib`, `seaborn`, `sklearn`, `pyswarm`, `joblib`

## File Structure

```
output_02/
├── models/
│   └── [Model Files]
├── plots/
│   └── [Visualization Files]
└── results/
    └── [Performance and Metric Files]
```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/username/repository-name.git
   cd repository-name
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your dataset (`UHPC_Data.csv`) in the project folder or adjust the path in the code where the dataset is loaded.
2. Run the main Python script (`model_training.py`) to preprocess the data, optimize hyperparameters using PSO, and train various regression models.
3. The optimized models and performance metrics will be saved in the `output_02/` folder.

## Results

The model evaluates different algorithms such as:
- Random Forest Regressor
- Support Vector Regressor (SVR)
- Gradient Boosting Regressor
- Multi-layer Perceptron (Neural Network)
- K-Nearest Neighbors (KNN)

Each model is trained and optimized, and performance metrics including Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R², and Mean Absolute Percentage Error (MAPE) are calculated.

### Performance Metrics Comparison
The performance of all models is compared using RMSE and visualized in a bar chart.

![model_performance_comparison](https://github.com/user-attachments/assets/6bbcb7f5-b355-4971-a65a-5670e053909d)

## Acknowledgments

- **References:**
  - PSO optimization technique for hyperparameter tuning was implemented using the `pyswarm` library.
  - The dataset for UHPC was sourced from [source name or link, if applicable].
  
- **Creative Commons Zero v1.0 Universal (CC0 1.0)**: This repository is licensed under the CC0 1.0 Universal (CC0 1.0) Public Domain Dedication. You may copy, modify, distribute, and perform the work, even for commercial purposes, without asking permission.

## Future Work

- **Expand parameter spaces for hyperparameter optimization**: To further improve model performance, additional parameters can be added to the optimization process.
- **Integrate additional models**: Future work will involve incorporating models like XGBoost and CatBoost for comparison and better accuracy.
- **Feature importance analysis**: We plan to include feature importance analysis for each model to provide better interpretability of results.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a feature branch:  
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:  
   ```bash
   git commit -m 'Add some feature'
   ```
4. Push to the branch:  
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

## Contact

For questions, feedback, or contributions, please contact:

- Email: [rhudwill@gmail.com]
- GitHub: [https://github.com/Rod-Will]

---
