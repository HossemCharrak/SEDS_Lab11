import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from fpdf import FPDF
import matplotlib.pyplot as plt
import io

def generate_pdf_report(data, mse, future_predictions):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.set_font("Arial", size=16, style="B")
    pdf.cell(200, 10, txt="Bitcoin Price Analysis Report", ln=True, align='C')
    pdf.ln(10)

    # Descriptive Statistics
    pdf.set_font("Arial", size=12, style="B")
    pdf.cell(0, 10, "Descriptive Statistics", ln=True)
    pdf.set_font("Arial", size=10)
    stats = data.describe().to_string()
    pdf.multi_cell(0, 10, stats)
    pdf.ln(10)

    # Model Performance
    pdf.set_font("Arial", size=12, style="B")
    pdf.cell(0, 10, "Machine Learning Model Performance", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, f"Mean Squared Error (MSE): {mse:.2f}", ln=True)
    pdf.ln(10)

    # Future Predictions
    pdf.set_font("Arial", size=12, style="B")
    pdf.cell(0, 10, "Future Predictions", ln=True)
    pdf.set_font("Arial", size=10)
    for date, price in zip(future_predictions['Date'], future_predictions['Predicted Price']):
        pdf.cell(0, 10, f"{date}: ${price:.2f}", ln=True)

    # Visualization (save a chart as an image and embed)
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['close'], label="Bitcoin Price")
    plt.title("Bitcoin Price Trend")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig("price_chart.png")
    pdf.add_page()
    pdf.image("price_chart.png", x=10, y=20, w=180)

    return pdf.output(dest='S').encode('latin1')


def generate_excel_report(data, mse, future_predictions):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Write data overview
        data.to_excel(writer, sheet_name="Data Overview")
        # Write descriptive statistics
        data.describe().to_excel(writer, sheet_name="Statistics")
        # Write future predictions
        future_df = pd.DataFrame(future_predictions)
        future_df.to_excel(writer, sheet_name="Future Predictions", index=False)
        # Write model performance
        pd.DataFrame({"Metric": ["Mean Squared Error"], "Value": [mse]}).to_excel(writer, sheet_name="Model Performance", index=False)

    output.seek(0)
    return output

# Function to load Bitcoin price data from a CSV file
@st.cache_data
def load_bitcoin_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
    df = df.dropna(subset=['Date'])
    df.set_index('Date', inplace=True)
    return df

# Machine Learning Model Implementation
@st.cache_resource
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
    return model, mse, X_train, y_train, X_test, y_test, predictions

# Function to predict future prices
def predict_future_prices(model, last_price, days):
    future_prices = []
    for _ in range(days):
        next_price = model.predict([[last_price]])[0]
        future_prices.append(next_price)
        last_price = next_price
    return future_prices

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
            tabs = st.tabs(["Data Overview", "Visualizations", "Machine Learning", "Prediction"])

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
                model, mse, X_train, y_train, X_test, y_test, predictions = train_ml_model(filtered_data)

                # Display model performance
                st.write(f"Mean Squared Error: {mse:.2f}")

                # Prediction Visualization
                prediction_fig = go.Figure()
                prediction_fig.add_trace(go.Scatter(x=X_test.index, y=y_test, mode='lines', name='Actual'))
                prediction_fig.add_trace(go.Scatter(x=X_test.index, y=predictions, mode='lines', name='Predicted'))
                prediction_fig.update_layout(title="Actual vs Predicted Prices", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(prediction_fig)

            # Prediction Tab
            with tabs[3]:
                st.subheader("Future Price Prediction")
                days_to_predict = st.number_input("Enter the number of future days to predict:", min_value=1, max_value=30, value=7)
                
                # Predict future prices
                last_known_price = filtered_data['close'].iloc[-1]
                future_prices = predict_future_prices(model, last_known_price, days_to_predict)
                future_dates = pd.date_range(start=filtered_data.index[-1] + pd.Timedelta(days=1), periods=days_to_predict)

                # Display predictions
                prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_prices})
                st.write(prediction_df)

                # Visualization of predictions
                future_fig = px.line(prediction_df, x="Date", y="Predicted Price", title="Future Bitcoin Price Predictions", labels={"x": "Date", "Predicted Price": "Price"})
                st.plotly_chart(future_fig)
                st.subheader("Exportable Reports")

                # Generate PDF
                if st.button("Generate PDF Report"):
                    pdf_report = generate_pdf_report(filtered_data, mse, prediction_df)
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_report,
                        file_name="Bitcoin_Analysis_Report.pdf",
                        mime="application/pdf"
                    )

                # Generate Excel
                if st.button("Generate Excel Report"):
                    excel_report = generate_excel_report(filtered_data, mse, prediction_df)
                    st.download_button(
                        label="Download Excel Report",
                        data=excel_report,
                        file_name="Bitcoin_Analysis_Report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )