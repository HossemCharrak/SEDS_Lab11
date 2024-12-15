import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to load Bitcoin price data from a CSV file
def load_bitcoin_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
    df = df.dropna(subset=['Date'])
    df.set_index('Date', inplace=True)
    return df

# Machine Learning Model Implementation
def train_ml_model(data):
    data['Lagged_Price'] = data['close'].shift(1)
    data = data.dropna()
    X = data[['Lagged_Price']]
    y = data['close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    return model, mse, X_test, y_test, predictions

# Streamlit App
st.title("Bitcoin Price Analysis Dashboard")
st.sidebar.title("Options")

# File uploader
data_file = st.sidebar.file_uploader("Upload Bitcoin CSV File", type=["csv"])

if data_file:
    data = load_bitcoin_data(data_file)
    st.success("Data loaded successfully!")

    # Date range selection
    st.sidebar.subheader("Select Date Range")
    start_date = st.sidebar.date_input("Start Date", value=data.index.min())
    end_date = st.sidebar.date_input("End Date", value=data.index.max())

    # Ensure valid date range
    if start_date > end_date:
        st.error("Start Date must be before End Date.")
    else:
        filtered_data = data[(data.index >= pd.Timestamp(start_date)) & (data.index <= pd.Timestamp(end_date))]

        if filtered_data.empty:
            st.warning("No data available for the selected date range.")
        else:
            st.success("Data filtered successfully!")

            # Tabs for different analyses
            tabs = st.tabs(["Data Overview", "Visualizations", "Machine Learning"])

            # Data Overview Tab
            with tabs[0]:
                st.subheader("Descriptive Statistics")
                st.write(filtered_data.describe())

            # Visualization Tab
            with tabs[1]:
                st.subheader("Interactive Visualizations")
                # Price trend visualization
                fig = px.line(filtered_data, x=filtered_data.index, y="close", title="Bitcoin Price Trend", labels={"x": "Date", "close": "Bitcoin Price (USD)"})
                st.plotly_chart(fig)

                # Correlation heatmap (exclude non-numeric columns)
                numeric_data = filtered_data.select_dtypes(include=[np.number])  # Select only numeric columns
                numeric_data = numeric_data.dropna()  # Drop rows with NaN values
                corr_matrix = numeric_data.corr()  # Compute correlation matrix
                heatmap = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap")
                st.plotly_chart(heatmap)

            # Machine Learning Tab
            with tabs[2]:
                st.subheader("Machine Learning Model")
                model, mse, X_test, y_test, predictions = train_ml_model(filtered_data)

                # Display model performance
                st.write(f"Mean Squared Error: {mse:.2f}")

                # Prediction Visualization
                prediction_fig = go.Figure()
                prediction_fig.add_trace(go.Scatter(x=X_test.index, y=y_test, mode='lines', name='Actual'))
                prediction_fig.add_trace(go.Scatter(x=X_test.index, y=predictions, mode='lines', name='Predicted'))
                prediction_fig.update_layout(title="Actual vs Predicted Prices", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(prediction_fig)
else:
    st.info("Please upload a Bitcoin CSV file to begin.")
