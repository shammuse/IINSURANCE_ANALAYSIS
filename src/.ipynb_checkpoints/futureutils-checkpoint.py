import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class InsuranceDataUtils:
    def __init__(self, df, target_column):
        self.df = df.copy()
        self.target_column = target_column
        self.df_preprocessed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def add_date_features(self):
        # Extract year and month from TransactionMonth (assuming it's in 'YYYY-MM' format)
        self.df['TransactionYear'] = pd.to_datetime(self.df['TransactionMonth']).dt.year
        self.df['TransactionMonthOnly'] = pd.to_datetime(self.df['TransactionMonth']).dt.month

    def add_vehicle_age(self):
        # Calculate vehicle age based on the current year
        current_year = pd.Timestamp.now().year
        self.df['VehicleAge'] = current_year - self.df['RegistrationYear']

    def combine_province_zone(self):
        # Combine Province with MainCrestaZone into a new feature
        self.df['ProvinceZone'] = self.df['Province'].astype(str) + '_' + self.df['MainCrestaZone'].astype(str)

    def apply_log_transformation(self):
        # Ensure 'SumInsured' and 'CapitalOutstanding' are numeric
        self.df['SumInsured'] = pd.to_numeric(self.df['SumInsured'], errors='coerce')
        self.df['CapitalOutstanding'] = pd.to_numeric(self.df['CapitalOutstanding'], errors='coerce')

        # Replace NaN values with a small constant
        self.df['SumInsured'] = self.df['SumInsured'].fillna(1e-5)
        self.df['CapitalOutstanding'] = self.df['CapitalOutstanding'].fillna(1e-5)

        # Replace any remaining zero or negative values with a small constant
        self.df['SumInsured'] = self.df['SumInsured'].clip(lower=1e-5)
        self.df['CapitalOutstanding'] = self.df['CapitalOutstanding'].clip(lower=1e-5)

        # Apply log transformation
        self.df['LogSumInsured'] = np.log1p(self.df['SumInsured'])
        self.df['LogCapitalOutstanding'] = np.log1p(self.df['CapitalOutstanding'])

        # Verify results
        print(self.df[['LogSumInsured', 'LogCapitalOutstanding']].describe())

    def get_selected_features_df(self):
        # Select only the newly created features and the target variables
        selected_columns = [
            'TransactionYear', 'TransactionMonthOnly', 'VehicleAge', 
            'ProvinceZone', 'LogSumInsured', 'LogCapitalOutstanding', 
            'TotalPremium', 'TotalClaims'
        ]
        # Return a new dataframe with only the selected columns
        return self.df[selected_columns]

    def preprocess_features(self):
        print("Starting feature preprocessing...")  # Debug statement

        # Define columns
        categorical_features = ['ProvinceZone']
        numerical_features = ['VehicleAge', 'LogSumInsured', 'LogCapitalOutstanding', 'TotalPremium', 'TotalClaims']

        # Define preprocessors
        categorical_preprocessor = OneHotEncoder(drop='first', sparse_output=False)  # Updated argument name
        numerical_preprocessor = StandardScaler()

        # Combine preprocessors
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_preprocessor, numerical_features),
                ('cat', categorical_preprocessor, categorical_features)
            ]
        )
        
        try:
            # Apply preprocessing
            X_preprocessed = preprocessor.fit_transform(self.df)
            print("Preprocessing successful.")  # Debug statement
            
            # Get feature names after transformation
            feature_names = (
                numerical_features + 
                list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
            )
            
            # Convert to DataFrame
            self.df_preprocessed = pd.DataFrame(X_preprocessed, columns=feature_names)
            print("DataFrame created successfully with shape:", self.df_preprocessed.shape)  # Debug statement
        
        except Exception as e:
            print("Error during preprocessing:", e)  # Error handling

    def split_data(self, test_size=0.2, random_state=42):
        if self.df_preprocessed is None:
            raise ValueError("Preprocessed DataFrame has not been created. Call preprocess_features first.")
        
        if self.target_column not in self.df_preprocessed.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in the preprocessed DataFrame.")
        
        X = self.df_preprocessed.drop(columns=[self.target_column])
        y = self.df_preprocessed[self.target_column]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training data shape for {self.target_column}: {self.X_train.shape}")
        print(f"Testing data shape for {self.target_column}: {self.X_test.shape}")

    def get_train_test_data(self):
        if self.X_train is None or self.X_test is None:
            raise ValueError("Data has not been split. Call split_data first.")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_preprocessed_df(self):
        if self.df_preprocessed is None:
            raise ValueError("Preprocessed DataFrame has not been created. Call preprocess_features first.")
        return self.df_preprocessed

    def get_dataframe(self):
        # Return the full updated dataframe
        return self.df
