# 📧 Spam Detection Web App

An end-to-end **Machine Learning project** that detects whether a given message is **Spam** or **Not Spam**.  
The project includes **data cleaning, model training, hyperparameter optimization (Optuna)**, and a **Flask web interface** for real-time prediction.

---

## 🚀 Project Overview

This project demonstrates how to build a spam detection system from scratch, including:

- Cleaning and preprocessing text data  
- Building and evaluating multiple ML models  
- Optimizing hyperparameters using **Optuna**  
- Deploying the trained model in a **Flask web app**  
- Providing real-time spam detection with confidence scores  

---

## 🧠 Project Workflow

### 1. 🧹 Data Cleaning (`cleaning.ipynb`)
- Preprocessed the dataset by:
  - Lowercasing  
  - Removing punctuation, stopwords, and special characters  
  - Lemmatization  
- Saved the cleaned dataset for model training.

### 2. 🤖 Model Training (`model1.ipynb`)
- Converted text into numerical features using **TF-IDF Vectorization**  
- Trained multiple models:
  - Naive Bayes  
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
  - LightGBM  
- Evaluated models using **Accuracy**, **F1-score**, and **Classification Report**  
- Saved the best model as `best_f1_spam_model.pkl` using `pickle`.

### 3. ⚙️ Hyperparameter Optimization (`optuna.ipynb`)
- Used **Optuna** to automatically search for the best hyperparameters.  
- Improved model accuracy and reduced overfitting.

### 4. 🌐 Web Application (`app.py`)
- Built with **Flask** for simple deployment and real-time predictions.
- Routes:
  - `/` → Home page (user input form)
  - `/predict_json` → API endpoint (returns prediction + confidence)
  - `/ping` → Health check
- Uses the saved model (`best_f1_spam_model.pkl`) for inference.

---

## 🧩 Folder Structure
```
📁 Spam-Detection-App
│
├── cleaning.ipynb # Data cleaning and preprocessing
├── model1.ipynb # Model training and evaluation
├── optuna.ipynb # Hyperparameter optimization
├── app.py # Flask web app for deployment
├── models/
│ └── best_f1_spam_model.pkl # Trained model file
├── templates/
│ └── home.html # Frontend template
├── static/ # Optional: CSS, JS, etc.
├── requirement.txt # Python dependencies
└── README.md # Project documentation
```

---

## 🧰 Installation & Setup

### 🔹 Step 1: Clone the repository
```bash
git clone https://github.com/your-username/spam-detection-app.git
cd spam-detection-app
```
🔹 Step 2: Create a virtual environment
```
python -m venv venv
```
Activate it:
```
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```
🔹 Step 3: Install dependencies
```
pip install -r requirement.txt
```
🔹 Step 4: Run the Flask app
```
python app.py
```
Then open your browser and go to:
👉 http://127.0.0.1:5000/

📈 Model Performance
| Model               | Accuracy | F1-Score | Notes                 |
| ------------------- | -------- | -------- | --------------------- |
| Naive Bayes         | 0.95     | 0.94     | Fast baseline         |
| Logistic Regression | 0.97     | 0.96     | Balanced and reliable |
| XGBoost             | 0.98     | 0.97     | Best performing       |
| LightGBM            | 0.98     | 0.97     | Comparable to XGBoost |

👨‍💻 Author
🔗 LinkedIn:https://www.linkedin.com/in/abhyuday-755b07292/
