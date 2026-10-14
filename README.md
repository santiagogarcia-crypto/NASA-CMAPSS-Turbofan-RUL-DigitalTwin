# Dual-Platform Digital Twin: Turbofan Engine RUL Estimation (NASA C-MAPSS FD001)

An end-to-end MLOps research project developing a **Digital Twin** for aircraft engine Remaining Useful Life (RUL) prediction. Built upon the NASA C-MAPSS FD001 dataset, integrating machine learning, analytical data modeling, and an interactive macro/micro web dashboard.

---

## 📌 Key Architectural Highlights

* **Predictive Core:** Random Forest Regressor ($RMSE = 17.07$, $MAE = 12.09$ cycles).
* **Data Pipeline:** 
  * Removal of 7 zero-variance sensors ($S_1, S_5, S_6, S_{10}, S_{16}, S_{18}, S_{19}$).
  * Piece-wise linear RUL target clipping ($RUL_{max} = 125$ cycles).
* **Dual-Platform Architecture:**
  * **Analytical Layer:** Power BI Star Schema (`Modelo_Entidad_Relación.pbix`) & DAX operational metrics.
  * **Web Layer:** Python-Streamlit (`app.py`) + Plotly interactive dashboard featuring a Macro/Micro fleet visualization framework.

---

## 🚀 Macro/Micro Visualization Framework

The Streamlit web application provides a real-time decision-support system for aviation maintenance engineers:
1. **Macro View (Fleet Level):** Absolute Error distribution histogram assessing global MLOps model reliability.
2. **Micro View (Unit Diagnostics):** Individual engine telemetry tracking with a dynamic red star marker navigating coordinates $(X = \text{Engine ID}, Y = \text{Predicted RUL})$.

---

## 📂 Repository Contents

* `app.py`: Interactive Streamlit dashboard.
* `Modelo_Entidad_Relación.pbix`: Power BI Data Model and Star Schema.
* `Artículo_IEEE_Gemelo_Digital_Nicolas_Garcia.pdf`: Formal IEEE research article.
* `Sustentacion_y_defensa.pdf`: Academic defense statement and Macro/Micro framework exposition.
* `modelo_ia.py` & `cargar.py`: Machine Learning training pipeline and data processing scripts.

---

## 💻 Quick Start (Local Reproduction)

To run the Digital Twin dashboard on your local machine, follow these steps:

```bash
# 1. Clone the public repository
git clone [https://github.com/santiagogarcia-crypto/NASA-CMAPSS-Turbofan-RUL-DigitalTwin.git](https://github.com/santiagogarcia-crypto/NASA-CMAPSS-Turbofan-RUL-DigitalTwin.git)

# 2. Navigate into the project directory
cd NASA-CMAPSS-Turbofan-RUL-DigitalTwin

# 3. Install required Python packages
pip install -r requirements.txt

# 4. Launch the interactive dashboard
python -m streamlit run app.py
