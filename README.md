#  Task 3: Car Price Prediction with Machine Learning

##  Overview
This project predicts car selling prices based on various features such as vehicle age, mileage (kms driven), fuel type, transmission, seller type, and initial price using Machine Learning.

---

##  Tools & Libraries Used
- **Python**
- **Pandas** (Data Manipulation & Cleaning)
- **Scikit-learn** (Data Preprocessing, Model Training & Evaluation)
- **Matplotlib & Seaborn** (Data Visualization)

---

##  Workflow & Data Preprocessing
1. **Data Loading**: Imported `car data.csv` into a Pandas DataFrame.
2. **Feature Engineering**:
   - Calculated vehicle age from manufacturing year.
   - Applied One-Hot Encoding / Label Encoding to categorical features (`Fuel_Type`, `Seller_Type`, `Transmission`).
3. **Train-Test Split**: Divided dataset into training (80%) and testing (20%) sets.
4. **Model Training**: Trained regression model(s) (e.g., Linear Regression / Random Forest).
5. **Evaluation**: Evaluated performance using $R^2$ Score and Mean Absolute Error (MAE).

---

## ss How to Run
```bash
# Clone the repository
git clone [https://github.com/Muntaha-3/CodeAlpha_Task_3_Car_Price_Prediction.git](https://github.com/Muntaha-3/CodeAlpha_Task_3_Car_Price_Prediction.git)

# Navigate to directory
cd CodeAlpha_Task_3_Car_Price_Prediction

# Run the script
python main.py