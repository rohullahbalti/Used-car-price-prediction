# Car Price Prediction Project

## Overview
This project is a machine learning-based car price prediction system that estimates both current market prices and future prices (up to 5 years) for used cars. The system uses a comprehensive dataset from OLX Pakistan to train predictive models.

## Dataset
- **Source**: OLX Pakistan car listings (`OLX_cars_dataset00.csv`)
- **Size**: Contains thousands of car listings with detailed features
- **Features**: 
  - Make, Model, Year, KM's driven
  - Fuel type, Registration status, Car documents
  - Assembly type (Local/Imported), Transmission type
  - Price (target variable for current price prediction)

## Data Preprocessing
1. **Data Cleaning**: Removed duplicates and irrelevant columns
2. **Feature Engineering**:
   - Created `Age` feature (2024 - Year)
   - Binned `Year` into ranges (1999-2004, 2004-2008, etc.)
   - Binned `KM's driven` into appropriate ranges
3. **Feature Selection**: Removed irrelevant car models and future years (2024)

## Machine Learning Models

### 1. Current Price Prediction Model
- **Algorithm**: Random Forest Regressor
- **Preprocessing Pipeline**:
  - OneHotEncoding for categorical features (Make, Model, Fuel, etc.)
  - MinMaxScaler for numerical features (Year, KM's driven, Age, etc.)
  - QuantileTransformer for output distribution normalization
- **Performance**: RMSE of approximately 272,169 PKR

### 2. Future Price Prediction Model
- **Algorithm**: MultiOutput Random Forest Regressor (predicts 1-5 years ahead)
- **Price Projection**: Based on inflation rate (8%) and tax rate (1%)
- **Performance**: RMSE of approximately 367,243 PKR

## Implementation Features

### Interactive Interface
The project includes a Jupyter Notebook-based interactive interface with:
- Dropdown menus for car makes and models
- Sliders and input fields for car specifications
- Real-time price predictions
- Future price projections (1-5 years)

### Key Functionalities
1. **Current Price Prediction**: Estimates the current market value of a car
2. **Future Price Projection**: Predicts car values for the next 1-5 years
3. **User-Friendly Interface**: Easy input of car specifications through widgets

## File Structure
- `final_prediction_current_price.pkl` - Serialized current price model
- `rought2future.pkl` - Serialized future price model
- `OLX_cars_dataset00.csv` - Original dataset
- Jupyter Notebook with complete code and interactive interface

## Usage
1. Run the Jupyter Notebook
2. Use the interactive widgets to input car specifications:
   - Select car make and model
   - Enter year, kilometers driven
   - Select fuel type, transmission, etc.
3. Get instant price predictions for current and future values

## Technical Requirements
- Python 3.8+
- Scikit-learn, Pandas, NumPy
- Jupyter Notebook
- Joblib for model serialization

## Applications
- Used car valuation for buyers and sellers
- Investment analysis for car purchases
- Market trend analysis in automotive sector

The project demonstrates effective use of machine learning for real-world price prediction problems with a user-friendly interface for practical application.
