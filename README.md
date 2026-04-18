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
- Supervised models (Random Forest, XGBoost) achieved high performance (ROC-AUC ~0.98)  
- Unsupervised model (Isolation Forest) showed weak separation (ROC-AUC ~0.47)  
- Deep learning model (ANN) showed lower recall for attack detection  

## Key Insights
- Tree-based models performed best on structured network traffic data  
- Unsupervised and ANN-based approaches struggled due to overlapping patterns between normal and attack traffic  
- Traditional machine learning models outperformed deep learning for this tabular dataset  

## Tools & Technologies
Python, Pandas, Scikit-learn, Matplotlib  

## Main Script
The primary workflow of this project is organised in:
- `main_pipeline.py`

## Supporting Scripts
Detailed model-specific analysis and evaluation are available in the `notebook/` folder.

## Skills Demonstrated
Machine Learning, Deep Learning, Cybersecurity Analytics, Anomaly Detection, Model Evaluation, Feature Engineering
