# Climate CO₂ Emission Prediction – Midterm Project

## 📖 **Project Overview**

This project is part of the DataTalksClub Machine Learning Zoomcamp **Midterm Project**.  

Climate change is one of the most significant global challenges of our time. Carbon dioxide (CO₂) emissions are the primary driver of global warming, contributing to over 75% of the increase in global temperature.

By building a regression model based on climate indicators, this project helps:
- Governments and researchers analyze CO₂ trends
- Support faster climate mitigation policymaking
- Evaluate the effectiveness of renewable energy adoption
- Enhance public education and environmental awareness

The project covers:
- Data preparation  
- Feature engineering  
- Model training  
- Model selection  
- Model evaluation  
- Saving and loading models  
- Serving the model via an API  
- Containerization with Docker  

---

# 📌 **1. Problem Description**

CO₂ concentration is one of the most important indicators for understanding climate change.  
This project aims to develop a model that can **predict future CO₂ levels** using environmental variables such as:

- Temperature  
- Sea Level  
- Precipitation  
- Humidity  
- Wind Speed  
- CO₂_lag_1 (previous CO₂ value)  

The objective is to expose the end result as a **reproducible API service**, so the model can be queried programmatically.

---

# 🧠 **2. Model & Approach**

The following steps were implemented:

### **Data Processing**
- Loaded and cleaned historical climate data  
- Created a lag feature for CO₂  
- Normalized and prepared data for modelling  

### **Model Training**
Models tested:
- RandomForestRegressor  
- GradientBoostingRegressor  
- XGBoost  

The final chosen model:  
✅ **RandomForestRegressor** (best performance based on RMSE)

The model is saved as: models/best_model.pkl


---

# **3. How to Run the Project**

There are **two main ways** to use this project:

---

## **A. Run Locally (Without Docker)**

### **1. Clone the repository**
git clone <your-repo-url>
cd midterm_project

### **2. Install dependencies**
pip install -r requirements.txt

### **3. Run the FastAPI service**
uvicorn predict:app --reload

The API will run on:

http://127.0.0.1:8000

### **4. Test prediction**

Use the /predict endpoint:

POST → http://127.0.0.1:8000/predict

Example JSON body:

{
  "Temperature": 20,
  "Sea_Level_Rise": 0.5,
  "Precipitation": 40,
  "Humidity": 60,
  "Wind_Speed": 30,
  "CO2_lag_1": 400
}

Response:

{
  "prediction": 401.10
}

## **B. Run Using Docker (Recommended)**
### **1. Build the Docker image**
docker build -t climate-model-azzah .

### 2. **Run the container**
docker run -p 8000:8000 climate-model-azzah

API will be available at:

http://localhost:8000

---

# 🗂️ **Project Structure**

midterm_project_azzah/
│── data/
│    └── climate_change_data.csv
│── models/
│   └── best_model.pkl
│── results
│   └── feature_importance.csv
│── src/
│   ├── train_model.py
│   └── predict.py
│── Dockerfile
│── notebook.ipynb
│── README.md
│── requirements.txt


# **4. Training Script**

Training is handled by:
src/train_model.py

It:
- Loads data
- Trains models
- Evaluates performance
- Saves best model to models/best_model.pkl

Run manually:
python src/train_model.py

# **5. API Service**

The FastAPI app is located in:
predict.py

It exposes:
POST /predict → Returns CO₂ prediction


# **6. Requirements**

All dependencies are in:
requirements.txt

# **7. Conclusion**

This project demonstrates the full ML pipeline:
- Data preparation
- Training & model selection
- Serving via API
- Containerization

The final deliverable is a reproducible ML service that predicts CO₂ concentration.

### 🙌 **Credits**
Created by **Azzah Sumaiyah**

