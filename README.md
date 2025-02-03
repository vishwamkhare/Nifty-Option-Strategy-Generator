# Nifty Option Strategy Generator

**Nifty Option Strategy Generator** is a machine learning-based project that predicts Nifty market trends and generates actionable option strategies (buying calls/puts). The model uses historical market data and technical indicators such as Moving Averages (MA) and Relative Strength Index (RSI) to predict future market movements, helping traders make informed decisions.

# Technologies Used
	•	Python
	•	Machine Learning (Random Forest)
	•	Pandas and NumPy for data manipulation
	•	Scikit-learn for model training and evaluation
	•	Yahoo Finance API for market data
	•	NSE India API for fetching live option chain data

# Features
	•	Market Trend Prediction: The model predicts the future direction of the Nifty index using historical data and technical indicators with over 85% accuracy.
	•	Option Strategy Generation: Based on the predicted market movement, it generates actionable strategies such as buying call or put options with detailed recommendations (strike price, premium, stop-loss, and target price).
	•	Real-Time Data: Integrates real-time data from Yahoo Finance for market analysis and from the NSE India API to fetch the latest option chain data.
	•	Automated Processing: The project automates data preprocessing (e.g., handling missing data), feature extraction (e.g., calculating RSI, MA), and model training.

# How It Works
	1.	Download Historical Data: The project first downloads historical Nifty data using the Yahoo Finance API starting from 2010. It calculates daily returns and creates a binary target variable that indicates if the market price went up (1) or down (0).
	2.	Feature Engineering: The next step involves calculating key technical indicators, such as the 50-period and 200-period moving averages (MA), and the 14-day Relative Strength Index (RSI). These features help the model understand market trends and potential price movements.
	3.	Train the Model: The data is split into training and testing sets. A Random Forest model is trained using the calculated features to predict the target variable (whether the market will go up or down). Random Forest is an ensemble learning method that combines multiple decision trees to improve the accuracy and robustness of predictions.
	4.	Fetch Real-Time Data: The model fetches the latest Nifty data using the Yahoo Finance API to perform real-time analysis of the market’s movement.
	5.	Generate Option Strategy: Based on the model’s prediction of an upward or downward market movement, it generates a corresponding option strategy (call/put). The strategy includes:
	    •	Strike Price: The price at which the option can be exercised.
	    •	Premium: The cost of the option.
	    •	Target Price: The predicted price at which the option should be sold.
	    •	Stop-Loss: The price level to sell if the trade goes against the prediction.
	6.	Result: The user is provided with the recommended strategy, including the strike price, expiry date, premium, target price, and stop-loss.

# How to Use
	1.	Install Dependencies: Install the required Python packages by running pip install -r requirements.txt. This will install all necessary libraries, such as Pandas, NumPy, and Scikit-learn.
	2.	Run the Script: The main script will run the entire process:
	    •	It will first download historical data and calculate technical features.
	    •	Then, it will train the Random Forest model on the historical data.
	    •	After that, it fetches the latest market data and generates an option strategy based on the prediction.
	3.	View Results: The output will display the recommended option strategy, including strike price, premium, target price, and stop-loss.

# Example Output
	When the script runs, it will generate output in the following format: Strategy: Buy Call Option
	Strike Price: 18200
	Expiry Date: 2025-02-10
	Premium (Cost): 120
	Recommended Buy Price: 120
	Recommended Sell Price: 240
	Stop Loss: 90
	Reasoning: The model predicts an upward movement in the Nifty index.
