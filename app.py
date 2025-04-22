import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample

st.set_page_config(page_title="IPC Value simulation tool", layout="wide")
st.title("IPC Value simulation tool")

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'encoders' not in st.session_state:
    st.session_state.encoders = {}
if 'numeric_columns' not in st.session_state:
    st.session_state.numeric_columns = []
if 'unique_skus' not in st.session_state:
    st.session_state.unique_skus = []
if 'unique_packaging_lines' not in st.session_state:
    st.session_state.unique_packaging_lines = []
if 'X_encoded' not in st.session_state:
    st.session_state.X_encoded = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'column_names' not in st.session_state:
    st.session_state.column_names = {}

# Create tabs
tab1, tab2 = st.tabs(["Data Upload & Model Configuration", "Scenario Analysis & MCMC Simulation"])

# Function to validate the CSV file
def validate_csv(df):
    # Check if the dataframe has at least 7 columns
    if len(df.columns) < 7:
        st.error("The CSV file must have at least 7 columns")
        return False
    
    # Check if 'Packaging_Line' and 'SKU' columns exist
    if 'Packaging_Line' not in df.columns or 'SKU' not in df.columns:
        st.error("The CSV file must contain 'Packaging_Line' and 'SKU' columns")
        return False
    
    # Count numeric columns (excluding Packaging_Line and SKU)
    numeric_cols = [col for col in df.columns if col not in ['Packaging_Line', 'SKU'] and pd.api.types.is_numeric_dtype(df[col])]
    if len(numeric_cols) < 5:
        st.error("The CSV file must contain at least 5 numeric columns")
        return False
    
    return True

# Function to fit the linear model
def fit_model(df, target_column):
    # Separate the target and input variables
    y = df[target_column].values
    
    # Get numeric input columns (excluding the target)
    numeric_cols = [col for col in df.columns if col not in ['Packaging_Line', 'SKU', target_column] 
                    and pd.api.types.is_numeric_dtype(df[col])]
    
    # Save numeric columns for later use
    st.session_state.numeric_columns = numeric_cols
    
    # Get categorical columns
    categorical_cols = ['Packaging_Line', 'SKU']
    
    # Save unique values for later use
    st.session_state.unique_packaging_lines = df['Packaging_Line'].unique().tolist()
    st.session_state.unique_skus = df['SKU'].unique().tolist()
    
    # Create and fit one-hot encoders for categorical variables
    encoders = {}
    encoded_data = []
    
    for col in categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded = encoder.fit_transform(df[[col]])
        encoders[col] = encoder
        encoded_data.append(encoded)
    
    # Include numeric inputs
    X_numeric = df[numeric_cols].values
    encoded_data.append(X_numeric)
    
    # Combine all input features
    X_encoded = np.concatenate(encoded_data, axis=1)
    
    # Create feature names for later reference
    column_names = {}
    
    for col in categorical_cols:
        feature_names = encoders[col].get_feature_names_out([col])
        column_names[col] = feature_names.tolist()
    
    column_names['numeric'] = numeric_cols
    
    st.session_state.column_names = column_names
    
    # Fit the linear model
    model = LinearRegression()
    model.fit(X_encoded, y)
    
    # Save model and encoders for scenario analysis
    st.session_state.model = model
    st.session_state.encoders = encoders
    st.session_state.X_encoded = X_encoded
    st.session_state.y = y
    
    return model, X_encoded, y

# Function to run fast bootstrap simulation
def run_bootstrap_simulation(input_values, packaging_line, sku, target_column, n_samples=500):
    # Get the trained linear regression model
    model = st.session_state.model
    
    if model is None:
        st.error("Please fit a model in Tab 1 first!")
        return None
    
    # Prepare input data for prediction
    # Create one-hot encoding for selected categorical values
    packaging_line_encoded = st.session_state.encoders['Packaging_Line'].transform([[packaging_line]])
    sku_encoded = st.session_state.encoders['SKU'].transform([[sku]])
    
    # Combine with numeric inputs
    numeric_inputs = np.array([input_values])
    
    X_scenario = np.concatenate([packaging_line_encoded, sku_encoded, numeric_inputs], axis=1)
    
    # Base prediction with the current model
    base_prediction = model.predict(X_scenario)[0]
    
    # Get original training data
    X = st.session_state.X_encoded
    y = st.session_state.y
    
    # Calculate residuals from the original model
    predictions = model.predict(X)
    residuals = y - predictions
    residual_std = np.std(residuals)
    
    # Store bootstrap results
    bootstrap_results = []
    
    # Run bootstrap simulations
    for _ in range(n_samples):
        # Create bootstrap sample
        indices = np.random.choice(len(X), len(X), replace=True)
        X_bootstrap = X[indices]
        y_bootstrap = y[indices]
        
        # Fit a model on the bootstrap sample
        bootstrap_model = LinearRegression()
        bootstrap_model.fit(X_bootstrap, y_bootstrap)
        
        # Make a prediction for our scenario
        bootstrap_prediction = bootstrap_model.predict(X_scenario)[0]
        
        # Add noise based on residual standard deviation
        noisy_prediction = bootstrap_prediction + np.random.normal(0, residual_std)
        
        # Store the result
        bootstrap_results.append(noisy_prediction)
    
    return np.array(bootstrap_results)

# Tab 1: Data Upload and Model Configuration
with tab1:
    st.header("Upload Your CSV Data")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            
            if validate_csv(data):
                st.success("CSV file successfully loaded!")
                
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(data.head())
                
                # Column selection for target variable
                st.subheader("Select Target Variable")
                
                # Get numeric columns
                numeric_cols = [col for col in data.columns if col not in ['Packaging_Line', 'SKU'] 
                               and pd.api.types.is_numeric_dtype(data[col])]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Available Numeric Columns:")
                    for col in numeric_cols:
                        if st.toggle(f"Use {col} as target", key=f"toggle_{col}"):
                            st.session_state.target_column = col
                
                with col2:
                    st.write("Selected Target Variable:")
                    if st.session_state.target_column:
                        st.info(f"Target: {st.session_state.target_column}")
                    else:
                        st.warning("Please select a target variable")
                
                # Button to fit the model
                if st.session_state.target_column:
                    if st.button("Fit Linear Model"):
                        with st.spinner("Fitting model..."):
                            model, X, y = fit_model(data, st.session_state.target_column)
                            
                            # Display model success
                            st.success("Model fitted successfully!")
                            
                            # Display model coefficients
                            st.subheader("Model Summary")
                            st.write(f"R² Score: {model.score(X, y):.4f}")
                            
                            # Display model coefficients
                            col_names = []
                            for key, value in st.session_state.column_names.items():
                                if isinstance(value, list):
                                    col_names.extend(value)
                            
                            coef_df = pd.DataFrame({
                                'Feature': col_names,
                                'Coefficient': model.coef_
                            })
                            
                            st.write("Model Coefficients (Top 10 by Magnitude):")
                            st.dataframe(coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index).head(10))
                            
                            st.write(f"Intercept: {model.intercept_:.4f}")
        except Exception as e:
            st.error(f"Error processing the file: {e}")

# Tab 2: Scenario Analysis and MCMC Simulation
with tab2:
    st.header("Scenario Analysis with MCMC Simulation")
    
    if st.session_state.data is None or st.session_state.model is None:
        st.warning("Please upload data and fit a model in Tab 1 first!")
    else:
        st.subheader("Configure Scenarios")
        
        # Filters for Packaging_Line and SKU
        col1, col2 = st.columns(2)
        
        with col1:
            packaging_line = st.selectbox(
                "Select Packaging Line", 
                options=st.session_state.unique_packaging_lines
            )
        
        with col2:
            sku = st.selectbox(
                "Select SKU",
                options=st.session_state.unique_skus
            )
        
        # Input numeric values for two scenarios
        st.subheader("Configure Input Values for Scenarios")
        
        col1, col2 = st.columns(2)
        
        scenario1_inputs = []
        scenario2_inputs = []
        
        with col1:
            st.write("Scenario 1 Inputs")
            for col in st.session_state.numeric_columns:
                val = st.number_input(
                    f"{col} (Scenario 1)",
                    value=float(st.session_state.data[col].mean()),
                    key=f"s1_{col}"
                )
                scenario1_inputs.append(val)
        
        with col2:
            st.write("Scenario 2 Inputs")
            for col in st.session_state.numeric_columns:
                val = st.number_input(
                    f"{col} (Scenario 2)",
                    value=float(st.session_state.data[col].mean()),
                    key=f"s2_{col}"
                )
                scenario2_inputs.append(val)
        
        # Run simulation buttons
        col1, col2 = st.columns(2)
        
        with col1:
            sim1_button = st.button("Simulate Scenario 1")
        
        with col2:
            sim2_button = st.button("Simulate Scenario 2")
        
        # Results section
        st.subheader("Simulation Results")
        
        # Create placeholders for results
        if 'scenario1_results' not in st.session_state:
            st.session_state.scenario1_results = None
            
        if 'scenario2_results' not in st.session_state:
            st.session_state.scenario2_results = None
        
        # Run simulations when buttons are clicked
        if sim1_button:
            with st.spinner("Running bootstrap simulation for Scenario 1..."):
                results = run_bootstrap_simulation(
                    scenario1_inputs,
                    packaging_line,
                    sku,
                    st.session_state.target_column,
                    n_samples=500
                )
                
                if results is not None:
                    st.session_state.scenario1_results = results
                    st.success("Scenario 1 simulation completed!")
        
        if sim2_button:
            with st.spinner("Running bootstrap simulation for Scenario 2..."):
                results = run_bootstrap_simulation(
                    scenario2_inputs,
                    packaging_line,
                    sku,
                    st.session_state.target_column,
                    n_samples=500
                )
                
                if results is not None:
                    st.session_state.scenario2_results = results
                    st.success("Scenario 2 simulation completed!")
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("Scenario 1 Mean Output:")
            if st.session_state.scenario1_results is not None:
                mean1 = np.mean(st.session_state.scenario1_results)
                st.info(f"{mean1:.4f}")
            else:
                st.write("Not simulated yet")
        
        with col2:
            st.write("Scenario 2 Mean Output:")
            if st.session_state.scenario2_results is not None:
                mean2 = np.mean(st.session_state.scenario2_results)
                st.info(f"{mean2:.4f}")
            else:
                st.write("Not simulated yet")
        
        with col3:
            st.write("Percent Difference:")
            if st.session_state.scenario1_results is not None and st.session_state.scenario2_results is not None:
                mean1 = np.mean(st.session_state.scenario1_results)
                mean2 = np.mean(st.session_state.scenario2_results)
                
                # Calculate percent difference
                if mean1 != 0:
                    pct_diff = ((mean2 - mean1) / abs(mean1)) * 100
                    st.info(f"{pct_diff:.2f}%")
                else:
                    st.warning("Cannot calculate percent difference (division by zero)")
            else:
                st.write("Both scenarios must be simulated")
        
        # Visualization
        st.subheader("Visualization of Results")
        
        if st.session_state.scenario1_results is not None or st.session_state.scenario2_results is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if st.session_state.scenario1_results is not None:
                mean1 = np.mean(st.session_state.scenario1_results)
                sns.kdeplot(st.session_state.scenario1_results, fill=True, alpha=0.5, color="blue", label=f"Scenario 1 (Mean: {mean1:.4f})")
                ax.axvline(mean1, color="blue", linestyle="--")
            
            if st.session_state.scenario2_results is not None:
                mean2 = np.mean(st.session_state.scenario2_results)
                sns.kdeplot(st.session_state.scenario2_results, fill=True, alpha=0.5, color="red", label=f"Scenario 2 (Mean: {mean2:.4f})")
                ax.axvline(mean2, color="red", linestyle="--")
            
            ax.set_xlabel(f"{st.session_state.target_column} (Output)")
            ax.set_ylabel("Density")
            ax.set_title("Bootstrap Simulation Results: Distribution of Output Variable")
            ax.legend()
            
            st.pyplot(fig)
        else:
            st.write("Run at least one simulation to see results")

if __name__ == "__main__":
    st.sidebar.title("About")
    st.sidebar.info(
        "Upload CSV containing production KPIs, "
        "fit a linear model, and run MCMC simulations to "
        "compare different scenarios."
    )
    
    st.sidebar.title("Instructions")
    st.sidebar.markdown(
        """
        1. **Tab 1**: Upload your CSV file and select a target variable
        2. **Tab 1**: Fit the linear model
        3. **Tab 2**: Select Packaging Line and SKU values
        4. **Tab 2**: Configure input values for two scenarios
        5. **Tab 2**: Run simulations and compare results
        """
    )