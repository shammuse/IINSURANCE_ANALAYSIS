import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class InsuranceDataUtils:
    def __init__(self, df):
        """
        Initializes the InsuranceDataUtils with a DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("The input must be a pandas DataFrame.")
        self.df = df

    def descriptive_statistics(self):
        """
        Computes descriptive statistics for numerical features in the DataFrame.
        """
        # Select all numerical columns and calculate their descriptive statistics
        numerical_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        variability = self.df[numerical_columns].describe()
        return variability
    def data_structure(self):
        """
        Reviews the data types of each column to confirm if they are properly formatted.
        """
        # Return the data types of each column
        column_types = self.df.dtypes
        return column_types

    def handle_missing_values(self) -> pd.DataFrame:
        """
        Handles missing values in the DataFrame by:
        - Dropping columns with more than 50% missing values
        - Imputing missing values for numerical columns with the median
        - Filling specific columns with default values if necessary
        Returns
        pd.DataFrame: The cleaned DataFrame.
        """
        # Drop columns with more than 50% missing values
        threshold = len(self.df) * 0.5
        self.df = self.df.dropna(thresh=threshold, axis=1)

        # Impute missing values for categorical columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.df.loc[:, col] = self.df[col].fillna('Unknown')

        # Impute missing values for numerical columns
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            if col != 'NumberOfVehiclesInFleet':  # Special case for this column
                self.df.loc[:, col] = self.df[col].fillna(self.df[col].median())

        # Impute missing values for the column 'NumberOfVehiclesInFleet' with 0
        if 'NumberOfVehiclesInFleet' in self.df.columns:
            self.df.loc[:, 'NumberOfVehiclesInFleet'] = self.df['NumberOfVehiclesInFleet'].fillna(0)

        return self.df
    def univariate_analysis(self):
        # Determine the number of plots needed
        num_numerical = len(self.df.select_dtypes(include=['float64', 'int64']).columns)
        num_categorical = len(self.df.select_dtypes(include=['object']).columns)
        
        # Create a figure with subplots arranged in 4 columns
        num_plots = num_numerical + num_categorical
        num_rows = (num_plots + 3) // 4  # Calculate the number of rows needed
        fig, axes = plt.subplots(num_rows, 4, figsize=(20, num_rows * 5), constrained_layout=True)
        
        # Flatten axes array for easy iteration
        axes = axes.flatten()

        # Plot histograms for numerical columns
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        for i, col in enumerate(numerical_cols):
            axes[i].hist(self.df[col].dropna(), bins=30, edgecolor='black')
            axes[i].set_title(f'Histogram of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        
        # Plot bar charts for categorical columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for j, col in enumerate(categorical_cols, start=len(numerical_cols)):
            sns.countplot(data=self.df, x=col, ax=axes[j])
            axes[j].set_title(f'Bar Chart of {col}')
            axes[j].set_xlabel(col)
            axes[j].set_ylabel('Count')
        
        # Hide unused axes
        for ax in axes[num_plots:]:
            ax.set_visible(False)
        
        plt.show()

    def bivariate_analysis(self):
        """
        Performs bivariate analysis by exploring relationships between TotalPremium
        and TotalClaims as a function of ZipCode.
        Also generates a correlation matrix for TotalPremium and TotalClaims.
        """
        # Scatter plot: TotalPremium vs TotalClaims, color-coded by ZipCode
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        sns.scatterplot(x='TotalPremium', y='TotalClaims', hue='ZipCode', data=self.df, palette='viridis')
        plt.title('TotalPremium vs TotalClaims by ZipCode')
        plt.legend(title='ZipCode', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Correlation matrix
        plt.subplot(1, 2, 2)
        corr = self.df[['TotalPremium', 'TotalClaims']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix: TotalPremium & TotalClaims')

        plt.tight_layout()
        plt.show()
    def bivariate_analysis(self):
        """
        Analyzes relationships between monthly changes in TotalPremium and TotalClaims
        using scatter plots and correlation matrices.
        """
        if 'TransactionMonth' not in self.df.columns:
            raise ValueError("DataFrame must contain a 'TransactionMonth' column.")
        
        # Rename 'TransactionMonth' to 'Date' and ensure it's in datetime format
        self.df.rename(columns={'TransactionMonth': 'Date'}, inplace=True)
        self.df['Date'] = pd.to_datetime(self.df['Date'])

        # Remove duplicate rows by aggregating data
        self.df = self.df.groupby('Date').agg({
            'TotalPremium': 'sum',
            'TotalClaims': 'sum'
        }).reset_index()

        # Set 'Date' as index
        self.df.set_index('Date', inplace=True)
        
        # Resample data by month using Month End frequency
        monthly_data = self.df.resample('ME').agg({
            'TotalPremium': 'sum',
            'TotalClaims': 'sum'
        })
        
        # Fill missing values using forward fill
        monthly_data.ffill(inplace=True)

        # Calculate monthly changes
        monthly_data['MonthlyTotalPremiumChange'] = monthly_data['TotalPremium'].pct_change()
        monthly_data['MonthlyTotalClaimsChange'] = monthly_data['TotalClaims'].pct_change()
        
        # Reset index to bring 'Date' back as a column
        monthly_data.reset_index(inplace=True)
        
        # Check for missing values after forward fill
        missing_values = monthly_data[['MonthlyTotalPremiumChange', 'MonthlyTotalClaimsChange']].isnull().sum()
        if missing_values.any():
            print("Missing values found in MonthlyTotalPremiumChange or MonthlyTotalClaimsChange after filling:")
            print(missing_values)
        
        # Plot scatter plots
        plt.figure(figsize=(10, 5))
        plt.scatter(monthly_data['MonthlyTotalPremiumChange'], monthly_data['MonthlyTotalClaimsChange'])
        plt.title('Monthly Total Premium vs. Total Claims Change')
        plt.xlabel('Monthly Total Premium Change')
        plt.ylabel('Monthly Total Claims Change')
        plt.show()

        # Calculate and print correlation matrix
        if not monthly_data[['MonthlyTotalPremiumChange', 'MonthlyTotalClaimsChange']].empty:
            corr_matrix = monthly_data[['MonthlyTotalPremiumChange', 'MonthlyTotalClaimsChange']].corr()
            print("Correlation Matrix:")
            print(corr_matrix)
        else:
            print("No data available for correlation calculation.")
    def preprocess_data(self):
        """
        Preprocesses the DataFrame by renaming columns, converting date columns, 
        and aggregating data.
        """
        # Check and rename 'TransactionMonth' to 'Date'
        if 'TransactionMonth' not in self.df.columns:
            raise ValueError("DataFrame must contain a 'TransactionMonth' column.")
        
        self.df.rename(columns={'TransactionMonth': 'Date'}, inplace=True)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Aggregating data
        self.df = self.df.groupby('Date').agg({
            'TotalPremium': 'sum',
            'TotalClaims': 'sum'
        }).reset_index()

        # Set 'Date' as index
        self.df.set_index('Date', inplace=True)
        
        # Resample data by month using Month End frequency
        monthly_data = self.df.resample('ME').agg({
            'TotalPremium': 'sum',
            'TotalClaims': 'sum'
        })
        
        # Fill missing values using forward fill
        monthly_data.ffill(inplace=True)

        # Check for missing values after forward fill
        missing_values = monthly_data[['TotalPremium', 'TotalClaims']].isnull().sum()
        if missing_values.any():
            print("Missing values found after filling:")
            print(missing_values)

        # Calculate monthly changes
        monthly_data['MonthlyTotalPremiumChange'] = monthly_data['TotalPremium'].pct_change()
        monthly_data['MonthlyTotalClaimsChange'] = monthly_data['TotalClaims'].pct_change()
        
        # Check for missing values in changes
        missing_changes = monthly_data[['MonthlyTotalPremiumChange', 'MonthlyTotalClaimsChange']].isnull().sum()
        if missing_changes.any():
            print("Missing values found in MonthlyTotalPremiumChange or MonthlyTotalClaimsChange after calculating changes:")
            print(missing_changes)
        
        # Reset index to bring 'Date' back as a column
        monthly_data.reset_index(inplace=True)
        
        # Debug: Print column names to ensure changes are added
        print("Columns after preprocessing:")
        print(monthly_data.columns)

        self.df = monthly_data

    def compare_data(self):
        """
        Compares trends in insurance cover type, premium, etc., across geographic regions using the 'Province' column.
        """
        # Strip any leading or trailing spaces from column names
        self.df.columns = self.df.columns.str.strip()

        # Verify that 'Province' column exists
        if 'Province' not in self.df.columns:
            raise ValueError("DataFrame must contain a 'Province' column.")
        
        # Verify the data type of 'Province' column
        if self.df['Province'].dtype != 'object':
            raise ValueError("'Province' column must be of type 'object' (string).")

        # Ensure 'Date' is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(self.df['Date']):
            self.df['Date'] = pd.to_datetime(self.df['Date'])

        # Aggregating total premiums by 'Province' and 'Date'
        geo_trends = self.df.groupby(['Province', pd.Grouper(freq='M')])['TotalPremium'].sum().unstack()

        # Plotting trends
        plt.figure(figsize=(12, 6))
        for province in geo_trends.columns:
            plt.plot(geo_trends.index, geo_trends[province], label=province)

        plt.title('Trends in Total Premiums by Province')
        plt.xlabel('Month')
        plt.ylabel('Total Premium')
        plt.legend()
        plt.grid(True)
        plt.show()

    def detect_outliers(self):
        """
        Uses box plots to detect outliers in numerical data.
        """
        numeric_columns = self.df.select_dtypes(include=['number']).columns

        plt.figure(figsize=(14, 7))
        for i, column in enumerate(numeric_columns, 1):
            plt.subplot(3, 4, i)
            sns.boxplot(y=self.df[column])
            plt.title(f'Boxplot of {column}')

        plt.tight_layout()
        plt.show()

    def visualize_data(self):
        """
        Produces 3 creative and beautiful plots that capture key insights.
        """
        # Ensure necessary columns exist
        required_columns = ['TotalPremium', 'TotalClaims', 'MonthlyTotalPremiumChange', 'MonthlyTotalClaimsChange']
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

        # Plot: Monthly Total Premiums and Monthly Total Claims on the same graph
        plt.figure(figsize=(12, 6))
        plt.plot(self.df['Date'], self.df['TotalPremium'], label='Total Premium', color='blue')
        plt.plot(self.df['Date'], self.df['TotalClaims'], label='Total Claims', color='red')
        plt.title('Monthly Total Premiums and Total Claims')
        plt.xlabel('Month')
        plt.ylabel('Amount')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot 2: Premium vs Claims Change
        plt.figure(figsize=(10, 5))
        plt.scatter(self.df['MonthlyTotalPremiumChange'], self.df['MonthlyTotalClaimsChange'], alpha=0.5)
        plt.title('Monthly Change: Premium vs Claims')
        plt.xlabel('Monthly Premium Change')
        plt.ylabel('Monthly Claims Change')
        plt.grid(True)
        plt.show()
