# Sri Lanka Vehicle Price Predictor

This is an end-to-end Machine Learning project to predict vehicle prices in Sri Lanka using real-world data scraped from ikman.lk vehicle advertisements. The project uses **XGBoost (XGBoost Regressor)** as the selected machine learning algorithm and includes **hyperparameter tuning with Optuna**, **model explainability using SHAP**, and a **Streamlit front-end** for users to make predictions and view explanations.

This project demonstrates the complete machine learning workflow including data scraping, preprocessing, model training, evaluation, explainability and front-end integration.

## Directory Structure

- `app/`: Contains the Streamlit web application  
- `models/`: Saved trained models and encoders  
- `outputs/plots/`: Saved evaluation reports and visualizations (**Feature Importance**, **SHAP plots**, etc.)  
- `src/`: Machine learning pipeline scripts (**preprocessing**, **training**, **evaluation**, **explainability**)  
- `data/`: Processed datasets  
- `scrape.py`: Script to scrape vehicle ads from ikman.lk  
- `requirements.txt`: Python dependencies  
- `README.md`: Project documentation  

## Setup Instructions

### 1. Install Dependencies
Ensure **Python 3.8+** is installed. Run the following command in the project root:
```bash
pip install -r requirements.txt