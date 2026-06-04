> **MSc Data Science Dissertation Project — Coventry University (2025)**

# Anomaly Detection in Network Traffic

This project focuses on detecting anomalous behaviour in large-scale network traffic data using machine learning and deep learning techniques.

## Objective
To compare supervised, unsupervised and deep learning approaches for anomaly detection in cybersecurity data.

## Dataset
CICIDS2018 dataset

## Models Used
- Isolation Forest  
- Random Forest  
- XGBoost  
- Artificial Neural Network (ANN)  

## Workflow
- Data preprocessing  
- Feature engineering  
- Model training  
- Performance evaluation  
- Comparative analysis  

## Results
## Results Summary

| Model | Accuracy | ROC-AUC | F1 Score |
|---|---|---|---|
| Random Forest | 99.5% | 0.998 | 0.98 |
| XGBoost | 99.4% | 0.997 | 0.97 |
| Deep Learning ANN | 98.1% | 0.985 | 0.96 |
| Isolation Forest | - | 0.47 | 0.32 |  

## Key Insights
- Segmented customers into 3 distinct groups based on RFM (Recency, Frequency, Monetary) features
- Detected ~5% anomalous transactions using Isolation Forest
- Insights support targeted marketing and fraud prevention strategies  

## Tools & Technologies
Python, Pandas, Scikit-learn, Matplotlib
XGBoost, TensorFlow, Seaborn

## Main Script
The primary workflow of this project is organised in:
- `main_pipeline.py`

## Supporting Scripts
Detailed model-specific analysis and evaluation are available in the `notebook/` folder.

## Skills Demonstrated
Machine Learning, Deep Learning, Cybersecurity Analytics, Anomaly Detection, Model Evaluation, Feature Engineering
