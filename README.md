# 🏦 Loan Approval Prediction using Machine Learning

This project predicts whether a loan application will be approved or not using a machine learning model trained on historical loan data. The application also includes an interactive frontend built with **Streamlit**.

---

## 📂 Project Structure

```
loan-prediction-app/
│
├── train_model.py           # Trains the ML model and saves it as model.pkl
├── streamlit_app.py         # Streamlit UI for prediction
├── train_u6lujuX_CVtuZ9i.csv# Training dataset
├── requirements.txt         # Python dependencies
└── model.pkl                # Saved ML model with encoders
```

---

## 🚀 Features

- Cleaned and preprocessed dataset
- Random Forest Classifier for robust predictions
- Label Encoding of categorical values
- Evaluation using accuracy and classification report
- Saved model using `pickle`
- Streamlit frontend for real-time loan prediction

---

## 🔧 How to Run (Linux / Windows)

### 1. Clone the Repository
```bash
git clone <your-repo-link>
cd loan-prediction-app
```

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python train_model.py
```

### 5. Run the Streamlit App
```bash
streamlit run streamlit_app.py
```

---

## 🧪 Sample Inputs

**Approved Case:**
- Gender: Male
- Married: Yes
- Dependents: 0
- Education: Graduate
- Self Employed: No
- Applicant Income: 5000
- Coapplicant Income: 2000
- Loan Amount: 150
- Loan Term: 360
- Credit History: 1.0
- Property Area: Urban

**Not Approved Case:**
- Gender: Female
- Married: No
- Dependents: 3+
- Education: Not Graduate
- Self Employed: Yes
- Applicant Income: 1000
- Coapplicant Income: 0
- Loan Amount: 300
- Loan Term: 180
- Credit History: 0.0
- Property Area: Rural

---

## 📊 Dataset

- Source: [Analytics Vidhya](https://datahack.analyticsvidhya.com)
- CSV File: `train_u6lujuX_CVtuZ9i.csv`

---

## 📚 License

This project is for educational purposes and is licensed under the MIT License.

---

## 👨‍💻 Author

**Praneeth Ram**  
🎓 B.E. Computer Science, Annamalai University  
🔒 Aspiring Cybersecurity Analyst & ML Engineer  
🌐 [LinkedIn](https://www.linkedin.com/) (Update with your link)
