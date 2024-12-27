import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency
from scipy.stats import f_oneway

class InsuranceDataUtils:
    def __init__(self, df):
        """
        Initializes the InsuranceDataUtils with a DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("The input must be a pandas DataFrame.")
        self.df = df.copy()
        # A/B Hypothesis Testing
    def categorize_provinces(self, metric_col):
        """
        Categorize provinces into high-risk and low-risk based on the average value of a given metric.

        Parameters:
        - metric_col: The column name of the metric to use for categorization ('TotalClaims').

        Returns:
        - high_risk_provinces: List of high-risk provinces.
        - low_risk_provinces: List of low-risk provinces.
        """
        # Calculate average metric per province
        province_avg = self.df.groupby('Province')[metric_col].mean().reset_index()
        province_avg.columns = ['Province', 'AverageMetric']
        
        # Calculate the overall average metric
        overall_avg = province_avg['AverageMetric'].mean()
        
        # Define risk groups
        high_risk_provinces = province_avg[province_avg['AverageMetric'] > overall_avg]['Province'].tolist()
        low_risk_provinces = province_avg[province_avg['AverageMetric'] <= overall_avg]['Province'].tolist()
        
        return high_risk_provinces, low_risk_provinces

    def test_risk_differences(self, metric_col, high_risk_provinces, low_risk_provinces):
        """
        Test for significant differences in risk between high-risk and low-risk provinces.

        Parameters:
        - metric_col: The column name of the metric to analyze (e.g., 'TotalClaims').
        - high_risk_provinces: List of provinces classified as high-risk.
        - low_risk_provinces: List of provinces classified as low-risk.

        Returns:
        - t_statistic: The t-statistic of the test.
        - p_value: The p-value of the test.
        - interpretation: Interpretation of the results.
        """
        # Filter data for each group
        group_a_data = self.df[self.df['Province'].isin(high_risk_provinces)][metric_col]
        group_b_data = self.df[self.df['Province'].isin(low_risk_provinces)][metric_col]
        
        # Check if there is enough data in each group
        if len(group_a_data) == 0 or len(group_b_data) == 0:
            raise ValueError("One or both of the groups have no data. Please check the group names and data.")
        
        # Perform t-test
        t_statistic, p_value = stats.ttest_ind(group_a_data, group_b_data, equal_var=False)
        
        # Interpretation
        if p_value < 0.05:
            interpretation = "Reject the null hypothesis: There are significant differences in risk."
        else:
            interpretation = "Fail to reject the null hypothesis: No significant differences in risk."
        
        return t_statistic, p_value, interpretation
    
    def print_summary(self, t_statistic, p_value, interpretation):
        """
        Print the summary of the test results.
        """
        print(f"T-statistic: {t_statistic:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Interpretation: {interpretation}")


    def categorize_postal_codes(self, threshold=10):
        """
        Categorize postal codes based on their frequency.
        Postal codes with occurrences below the threshold are categorized as 'Other'.
        """
        
        # Count occurrences of each postal code
        postal_code_counts = self.df['PostalCode'].value_counts()
        
        # Define a category for postal codes that occur less than the threshold
        low_frequency_postal_codes = postal_code_counts[postal_code_counts < threshold].index
        
        # Use .loc to avoid SettingWithCopyWarning
        self.df.loc[:, 'PostalCodeCategory'] = self.df['PostalCode'].apply(
            lambda x: 'Other' if x in low_frequency_postal_codes else x
        )
        return self.df

    def analyze(self, threshold=10):
        """
        Perform the analysis by categorizing postal codes and running Chi-Square test.
        """
        # Step 1: Categorize Postal Codes
        self.categorize_postal_codes(threshold)
        
        # Step 2: Aggregate total claims by postal code category
        category_claims = self.df.groupby('PostalCodeCategory')['TotalClaims'].sum().reset_index()
        
        # Print the aggregated results for review
        print("Category Claims:")
        print(category_claims)
        
        # Step 3: Perform Chi-Square Test
        # Create a contingency table
        contingency_table = pd.crosstab(self.df['PostalCodeCategory'], self.df['TotalClaims'])
        
        # Perform Chi-Square Test
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Print Chi-Square Test Results
        print("Chi-Square Test Results:")
        print({
            'Chi-Square Statistic': chi2_stat,
            'P-Value': p_value,
            'Degrees of Freedom': dof,
            'Expected Frequencies': expected
        })
        
        # Interpretation
        if p_value < 0.05:
            interpretation = "Reject the null hypothesis: Significant differences in risk based on postal codes."
        else:
            interpretation = "Fail to reject the null hypothesis: No significant differences in risk based on postal codes."
        
        print(f"Interpretation: {interpretation}")

    def analyze_margins_by_postal_code(self, threshold=10):
        """
        Analyze margins (profit) by postal code categories."""
        
        # Step 1: Categorize postal codes
        self.categorize_postal_codes(threshold)
        
        # Step 2: Calculate margin
        self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']
        
        # Step 3: Aggregate margins by postal code category
        category_margin = self.df.groupby('PostalCodeCategory')['Margin'].mean().reset_index()
        
        # Print aggregated margins for review
        print("Category Margins: ")
        print(category_margin)
        
        # Prepare data for statistical tests
        grouped_data = [group['Margin'].values for name, group in self.df.groupby('PostalCodeCategory')]
        
        # Step 4: Perform ANOVA test
        f_stat, p_value = f_oneway(*grouped_data)
        
        # Print test results
        print("ANOVA Test Results:")
        print({
            'F-Statistic': f_stat,
            'P-Value': p_value
        })
        
        # Interpretation
        if p_value < 0.05:
            interpretation = "Reject the null hypothesis: Significant differences in margin between postal codes."
        else:
            interpretation = "Fail to reject the null hypothesis: No significant differences in margin between postal codes."
        
        print(f"Interpretation: {interpretation}")
    def preprocess_and_calculate(self):
        """
        Filter out rows where Gender is 'Not specified' or 'Unknown',
        and calculate average TotalClaims by Gender.
        """
        # Filter out unwanted rows
        self.df = self.df[~self.df['Gender'].isin(['Not specified', 'Unknown'])]
        
        # Calculate average TotalClaims by Gender
        self.avg_claims = self.df.groupby('Gender')['TotalClaims'].mean()
        return self.avg_claims

    def perform_t_test_and_interpret(self):
        """
        Perform a Two-Sample T-Test between 'Female' and 'Male' for TotalClaims,
        and print results and interpretation.
        """
        # Ensure that the average claims have been calculated
        if not hasattr(self, 'avg_claims'):
            raise ValueError("Average claims not calculated. Please run preprocess_and_calculate first.")
        
        # Split data into two groups
        male_claims = self.df[self.df['Gender'] == 'Male']['TotalClaims']
        female_claims = self.df[self.df['Gender'] == 'Female']['TotalClaims']
        
        # Perform T-Test
        t_stat, p_value = stats.ttest_ind(male_claims, female_claims, equal_var=False)
        
        # Print results
        print("Average Claims by Gender:")
        print(self.avg_claims)
        
        print("T-Test Results:")
        print({
            'T-Statistic': t_stat,
            'P-Value': p_value
        })
        
        # Interpretation
        if p_value < 0.05:
            interpretation = "Reject the null hypothesis: Significant differences in risk between genders."
        else:
            interpretation = "Fail to reject the null hypothesis: No significant differences in risk between genders."
        
        print(f"Interpretation: {interpretation}")
