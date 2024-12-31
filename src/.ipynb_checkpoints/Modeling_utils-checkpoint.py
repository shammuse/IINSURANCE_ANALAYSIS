import pandas as pd
import numpy as np
import shape
import lime
import lime.lime_tabular
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

class InsuranceModeling:
    def __init__(self, X_train_premium, X_test_premium, y_train_premium, y_test_premium,
                 X_train_claims, X_test_claims, y_train_claims, y_test_claims):
        self.X_train_premium = X_train_premium
        self.X_test_premium = X_test_premium
        self.y_train_premium = y_train_premium
        self.y_test_premium = y_test_premium
        self.X_train_claims = X_train_claims
        self.X_test_claims = X_test_claims
        self.y_train_claims = y_train_claims
        self.y_test_claims = y_test_claims
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(),
            'XGBoost': xgb.XGBRegressor(),
            'Decision Tree': DecisionTreeRegressor()
        }
        self.results_premium = {}
        self.results_claims = {}
    
    def train_models(self):
        for name, model in self.models.items():
            print(f"Training {name} for TotalPremium...")
            model.fit(self.X_train_premium, self.y_train_premium)
            y_pred_premium = model.predict(self.X_test_premium)
            self.results_premium[name] = {
                'model': model,
                'RMSE': np.sqrt(mean_squared_error(self.y_test_premium, y_pred_premium)),
                'MAE': mean_absolute_error(self.y_test_premium, y_pred_premium)
            }
            
            print(f"Training {name} for TotalClaims...")
            model.fit(self.X_train_claims, self.y_train_claims)
            y_pred_claims = model.predict(self.X_test_claims)
            self.results_claims[name] = {
                'model': model,
                'RMSE': np.sqrt(mean_squared_error(self.y_test_claims, y_pred_claims)),
                'MAE': mean_absolute_error(self.y_test_claims, y_pred_claims)
            }
    
    def evaluate_models(self):
        print("\nEvaluation for TotalPremium:")
        for name, metrics in self.results_premium.items():
            print(f"{name}: RMSE = {metrics['RMSE']:.2f}, MAE = {metrics['MAE']:.2f}")
        
        print("\nEvaluation for TotalClaims:")
        for name, metrics in self.results_claims.items():
            print(f"{name}: RMSE = {metrics['RMSE']:.2f}, MAE = {metrics['MAE']:.2f}")
    
    def feature_importance_analysis(self):
        for name, model in self.models.items():
            print(f"\nFeature Importance for {name} (TotalPremium):")
            if hasattr(model, 'feature_importances_'):
                feature_importances = model.feature_importances_
                for i, importance in enumerate(feature_importances):
                    print(f"Feature {i}: {importance:.4f}")
            else:
                print("Feature importance not available for this model.")
            
            print(f"\nFeature Importance for {name} (TotalClaims):")
            if hasattr(model, 'feature_importances_'):
                feature_importances = model.feature_importances_
                for i, importance in enumerate(feature_importances):
                    print(f"Feature {i}: {importance:.4f}")
            else:
                print("Feature importance not available for this model.")

    def shap_analysis(self, model_name):
        model_premium = self.results_premium[model_name]['model']
        model_claims = self.results_claims[model_name]['model']
        
        # SHAP analysis for TotalPremium
        explainer_premium = shap.Explainer(model_premium, self.X_train_premium)
        shap_values_premium = explainer_premium(self.X_test_premium)
        shap.summary_plot(shap_values_premium, self.X_test_premium)
        
        # SHAP analysis for TotalClaims
        explainer_claims = shap.Explainer(model_claims, self.X_train_claims)
        shap_values_claims = explainer_claims(self.X_test_claims)
        shap.summary_plot(shap_values_claims, self.X_test_claims)

    def lime_analysis(self, model_name):
        model_premium = self.results_premium[model_name]['model']
        model_claims = self.results_claims[model_name]['model']
        
        # LIME analysis for TotalPremium
        explainer_premium = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train_premium.values,
            feature_names=self.X_train_premium.columns,
            mode='regression'
        )
        for i in range(len(self.X_test_premium)):
            exp = explainer_premium.explain_instance(self.X_test_premium.iloc[i].values, model_premium.predict)
            exp.show_in_notebook(show_table=True, show_all=False)
        
        # LIME analysis for TotalClaims
        explainer_claims = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train_claims.values,
            feature_names=self.X_train_claims.columns,
            mode='regression'
        )
        for i in range(len(self.X_test_claims)):
            exp = explainer_claims.explain_instance(self.X_test_claims.iloc[i].values, model_claims.predict)
            exp.show_in_notebook(show_table=True, show_all=False)
