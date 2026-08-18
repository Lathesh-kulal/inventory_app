# Inventory Forecasting App

A Flask-based web application that allows users to upload sales data and generate inventory forecasts. The app automatically cleans missing data and provides a detailed dashboard with forecasting results for multiple products/SKUs.

## Features

- **CSV Upload**: Upload your sales data in CSV format.
- **Data Cleaning**: Automatically handles missing values by filling them with the column median/mode.
- **Data Export**: Download the cleaned dataset as an Excel file for your records.
- **Forecasting Dashboard**: View a comprehensive dashboard showing forecasted sales, category breakdowns, top/bottom performing SKUs, and potential alerts for inventory management.
- **Multi-Product Support**: Handles forecasting for multiple SKUs simultaneously.

## Requirements

The application requires Python and the following packages:
- flask >= 3.0
- pandas >= 2.0
- numpy >= 1.26
- scikit-learn >= 1.4
- openpyxl (for Excel export)

## Installation

1. Clone the repository or navigate to the project directory.
2. It's recommended to create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   pip install openpyxl
   ```

## Usage

1. Run the Flask application:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to `http://127.0.0.1:5000`.
3. Upload a CSV file containing at least the following columns:
   - `Date`
   - `Sales`
   - `Is_Holiday`
4. View the generated forecast dashboard and download the cleaned data if needed.

## Sample Dataset

A sample dataset (`watch_brand_v2.csv`) is included in the root directory. You can use this file to test the application. It contains monthly sales data for various luxury watch SKUs with the following columns:
- `Date`: The month of the sales record.
- `SKU`: Unique identifier for the product.
- `Category`: The product category (e.g., Luxury Dive).
- `Sales`: Number of units sold.
- `Unit_Price`, `Cost_Price`, `Selling_Price`: Pricing information.
- `Discount_Pct`, `Is_Promotion`, `Is_Holiday`: Influencing factors for sales.
- `Current_Stock`, `Lead_Time_Days`: Inventory management fields.

## Project Structure

- `app.py`: The main Flask application and routing.
- `model.py`: Contains the forecasting models and logic (`run_multi_product_forecast`).
- `preprocessing.py`: Data cleaning utilities (`clean_data`).
- `templates/`: HTML templates for the application UI.
- `static/`: Static assets (CSS, JS, etc.).
- `uploads/`: Temporary directory for uploaded and cleaned files.
