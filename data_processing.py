import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        
    def load_data(self):
        """Load data and drop duplicates/missing."""
        self.data = pd.read_csv(self.filepath)
        self.data = self.data.drop_duplicates()
        self.data = self.data.dropna()
        return self.data
    
    def feature_engineering(self):
        """Create smart features as requested."""
        df = self.data.copy()
        
        # 1. Income-to-experience ratio
        # Avoid division by zero
        df['Income_to_Experience'] = df['MonthlyIncome'] / (df['TotalWorkingYears'].replace(0, 1))
        
        # 2. Promotion delay flag
        # If YearsSinceLastPromotion is greater than 3, flag as 1 else 0
        df['Promotion_Delay_Flag'] = (df['YearsSinceLastPromotion'] > 3).astype(int)
        
        # 3. Engagement score
        # Using JobInvolvement (1-4) and JobSatisfaction (1-4)
        df['Engagement_Score'] = df['JobInvolvement'] + df['JobSatisfaction']
        
        # 4. Work stress score
        # Using DistanceFromHome, OverTime, WorkLifeBalance
        # Overtime: Yes = 1, No = 0
        df['Overtime_Num'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
        # Normalize DistanceFromHome to a 1-4 scale roughly (max is around 29)
        df['Distance_Score'] = pd.qcut(df['DistanceFromHome'], q=4, labels=[1, 2, 3, 4]).astype(int)
        # Higher stress = higher overtime, higher distance score, lower work life balance
        df['Work_Stress_Score'] = df['Overtime_Num'] * 2 + df['Distance_Score'] + (5 - df['WorkLifeBalance'])
        
        # 5. Employee stability index
        # YearsAtCompany / NumCompaniesWorked
        df['Stability_Index'] = df['YearsAtCompany'] / (df['NumCompaniesWorked'].replace(0, 0.5))
        
        # 6. Manager relationship score
        # YearsWithCurrManager + RelationshipSatisfaction
        df['Manager_Relation_Score'] = df['YearsWithCurrManager'] + df['RelationshipSatisfaction']
        
        self.data = df
        return df

    def get_features_and_target(self, target_col='Attrition'):
        """Separate numeric/categorical features and target."""
        df = self.data.copy()
        
        # Ensure target is integer 0 and 1
        y = df[target_col].astype(int)
        X = df.drop(columns=[target_col, 'Overtime_Num', 'Distance_Score'], errors='ignore')
        
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return X, y, num_cols, cat_cols
