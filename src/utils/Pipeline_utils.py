from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Calculate BMI
        df['BMI'] = df['Weight(kg)'] / (df['Height(m)'] ** 2)
        
        # Cholesterol ratio
        df['CHOL_ratio'] = df['Total_Cholesterol(mmol/L)'] / df['HDL-Cholesterol(mmol/L)']

        # Age group categorization
        def age_group(age):
            if age < 18: return 1.0
            elif age < 40: return 2.0
            elif age < 60: return 3.0
            else: return 4.0
        df['Age_group'] = df['Age(year)'].apply(age_group)

        # BMI category
        def bmi_category(bmi):
            if bmi < 18.5: return 1.0  # Underweight
            elif bmi < 25: return 2.0  # Normal
            elif bmi < 30: return 3.0  # Overweight
            else: return 4.0           # Obese
        df['BMI_category'] = df['BMI'].apply(bmi_category)

        # High cholesterol flag
        df['High_cholesterol'] = (df['Total_Cholesterol(mmol/L)'] > 6.2).astype(int)
        
        # High HbA1c flag (diabetes threshold)
        df['High_HbA1c'] = (df['Glycohemoglobin(%)'] >= 6.5).astype(int)

        # Interaction term: BMI × HbA1c
        df['BMI_HbA1c'] = df['BMI'] * df['Glycohemoglobin(%)']

        # High insulin flag
        df['High_insulin'] = (df['Insulin (pmol/L)'] > 90).astype(int)

        # Cardiovascular risk score (cumulative)
        df['Cardio_risk'] = (
            (df['Age(year)'] > 60).astype(int) +
            (df['BMI'] > 30).astype(int) +
            (df['Total_Cholesterol(mmol/L)'] > 6.2).astype(int) +
            (df['High_blood_pressure?'] == 1).astype(int)
        )

        return df

