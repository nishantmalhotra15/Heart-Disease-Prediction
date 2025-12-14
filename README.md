

#  Heart Disease Prediction using Random Forest

This project builds a **Machine Learning model** to predict the likelihood of heart disease using the **Cleveland Heart Disease Dataset**.
Along with the prediction model, an **interactive Streamlit web app** and a **Hyperparameter Tuning dashboard** are developed for real-time experimentation.

---

---

##  **Project Structure**

```
Heart-Disease-Prediction-RF/
│
├── app.py                          # Main Streamlit prediction app
├── pages/
│   └── 1_Hyperparameter_Tuning.py  # Tuning dashboard (Streamlit multipage)
│
├── Heart_Disease.ipynb             # Jupyter notebook: EDA, preprocessing, model building
├── handling_outliers.ipynb         # Outlier treatment
│
├── Heart_disease_cleveland_new.csv # Dataset
│
├── best_model.pkl                  # Saved Random Forest model
├── scaler.pkl                      # Saved StandardScaler object
│
├── requirements.txt                # Dependencies for Streamlit Cloud
├── README.md                       # Project documentation
└── .streamlit/
    └── config.toml                 # App theme configuration
```


---

## 💻 **Running the Streamlit App**

### **Local Execution**

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens:
✔ Main prediction page
✔ Sidebar with **Hyperparameter Tuning** page

---

##  **Hyperparameter Tuning Dashboard**

The `pages/1_Hyperparameter_Tuning.py` page allows:

* Changing number of trees
* Max depth
* Min samples split
* Min samples leaf
* Criterion
* Bootstrap
* Max features

It displays updated metrics and feature importance instantly.

---

## 👨‍💻 **Tech Stack**

* **Python**
* **Pandas, NumPy**
* **Matplotlib, Seaborn**
* **Scikit-learn**
* **Streamlit**
* **Pickle**

---


## 🏁 **Conclusion**

This project successfully demonstrates a complete ML workflow for medical prediction tasks.
The model performs well, aligns with clinical expectations, and the Streamlit app enhances usability and interpretability.

---
