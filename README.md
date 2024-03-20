# The-Endless-Line

## Overview

This project, *The-Endless-Line*, focuses on analyzing the waiting times at the Port Aventura park. Through a comprehensive approach that includes data cleaning, merging, feature engineering, exploratory data analysis, modeling, forecasting, and dashboard visualization, we aim to provide insights into factors influencing wait times and offer predictions to improve visitor experiences.

[Dashboard Screenshot](https://github.com/Behachee/The-Endless-Line/assets/140748662/b9d92c8e-3beb-47d7-b54b-8707aa306bd5 "Dashboard Screenshot")

## Project Structure

### A. Data Preparation

#### I. Cleaning & Merging
Data from various sources including Weather, Client types & transactions, and Attraction attributes were merged to create a unified data repository. Two main files were created differing only in the granularity of data (15 min, and 1 day averages).

#### II. Feature Engineering
Key steps included 'robustization' to reduce noise, dropping days when the park was closed, and addressing non-stationarity with the ADF test. New categorical variables like SEASONALITY and DAY PERIOD were introduced for analysis.

### B. Exploratory Data Analysis (EDA)
Utilizing the cleaned and engineered data, we conducted an EDA to uncover key dynamics and correlations between variables. Insights are documented in the EDA notebook.

### C. Modelling
1. **Random Forest Model**: Achieved an error rate below 10%, making it our model of choice for estimating wait times.
2. **Feature Importance Analysis**: Conducted in 'Log.ipynb' to understand the impact of various features on wait times.

### D. Forecasting
Implemented using Prophet for future wait time predictions, as detailed in "Prophet.py".

### E. Dashboard
A user-friendly dashboard to visualize insights and predictions. Run **app.py** and follow the terminal instructions to access the web app.

## Getting Started

To explore our dashboard and insights:

1. Clone this repository:
    ```git clone https://github.com/Behachee/The-Endless-Line.git```
2. Install required dependencies:
    ```pip install -r requirements.txt```
3. Run the application:
    ```python app.py```
4. Open the provided web app URL in your browser.

## Contributions

We welcome contributions and suggestions. Feel free to fork the repository, make changes, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.



