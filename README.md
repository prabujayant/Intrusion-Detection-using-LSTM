# Intrusion Detection System (IDS) Using UNSW-NB15 Dataset

## Overview
This project focuses on building an Intrusion Detection System (IDS) leveraging machine learning and deep learning techniques. The IDS is designed to detect network intrusions using the UNSW-NB15 dataset. It utilizes models like LSTM (Long Short-Term Memory) to identify and classify network attacks in real-time.

The LSTM model excels at handling sequential data, making it highly effective for detecting anomalies in network traffic patterns. It achieves an accuracy of **93.00%** and an F1 score of **94.44%**, showcasing its capability to address evolving cyber threats.

---

## Dataset Overview
- **Dataset**: UNSW-NB15
- **Features**: 44
- **Training Samples**: 82,332
- **Testing Samples**: 175,341
- **Classes**: Normal (0) and Attack (1)
- **Dataset Size**: ~0.04 GB

The dataset includes a mix of normal and malicious traffic, making it an excellent benchmark for intrusion detection systems.

---

## Project Structure

### 1. Data Preprocessing
- **Handling Missing Values**: No missing values in the dataset.
- **Encoding**: Categorical features (e.g., `proto`, `service`, `state`) are encoded using Label Encoding.
- **Normalization**: Numerical features are scaled using MinMaxScaler.
- **Feature Removal**: Unnecessary features like `id` are removed.

### 2. Feature Engineering
- **One-Hot Encoding**: Applied to categorical features for better representation.
- **Feature Selection**: Feature importance analyzed using Mutual Information Scores.

### 3. Model Training
- **LSTM**: A deep learning model leveraging LSTM layers for sequence-based intrusion detection.
- **Random Forest**: A traditional machine learning model for classification.
- **BMM**: An unsupervised anomaly detection model.

### 4. Model Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- **Cross-Validation**: Ensures model robustness.
- **Threshold Optimization**: Enhances performance by identifying optimal thresholds.

---

## Results

### LSTM Model
- **Accuracy**: 93.00%
- **F1 Score**: 94.44%
- **Precision**: 94.12%
- **Recall**: 95.03%

### BMM Model
- **Average ROC-AUC**: 73.43%
- **Best Threshold**: -11.12
- **F1-Score (Best Threshold)**: 67.10%

### Key Findings
- The LSTM model outperforms traditional methods, achieving high accuracy and F1 score.
- Threshold optimization significantly improves model performance.
- The BMM model offers a balanced trade-off between precision and recall, making it effective for anomaly detection.

---

## How to Run the Code

### Prerequisites
- **Python**: 3.10 or later
- **Required Libraries**:
  ```bash
  pip install numpy pandas tensorflow scikit-learn matplotlib seaborn joblib
  ```

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/intrusion-detection-lstm.git
   cd intrusion-detection-lstm
   ```

2. **Download the Dataset**:
   - Download the UNSW-NB15 dataset from [Kaggle](https://www.kaggle.com).
   - Place the dataset in the `data/` directory.

3. **Run the Code**:
   ```bash
   python train_lstm.py
   ```

4. **View Results**:
   - Performance metrics and visualizations (e.g., ROC curve, training history) are saved in the `results/` directory.

---

## Dependencies
- **Python Libraries**:
  - `numpy`
  - `pandas`
  - `tensorflow`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `joblib`

---

## Future Work
- **Hyperparameter Tuning**: Optimize model parameters to improve performance.
- **Ensemble Models**: Combine LSTM with GRU or CNN to enhance detection capabilities.
- **Real-Time Deployment**: Implement models for real-time intrusion detection.
- **Attack-Specific Classification**: Extend the model to classify specific attack types (e.g., DoS, Reconnaissance).

---

## Acknowledgments
- The UNSW-NB15 dataset was created by the Australian Centre for Cyber Security (ACCS).
- Inspired by research papers and tutorials on network intrusion detection.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Contact
For questions, suggestions, or collaborations, feel free to reach out:

- **Email**: [your-email@example.com]
- **GitHub**: [your-github-username]
