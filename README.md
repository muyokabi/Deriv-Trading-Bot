Deriv Trading Bot: A Two-Part System

This project contains two versions of an automated trading bot designed to predict the last digit of a financial instrument's tick data on the Deriv platform. The core functionality is split between a front-end trading script and a back-end analytical engine.

1. System Architecture: main.py & model.py
This version uses two separate Python scripts that communicate via a WebSocket connection.

Core Components
main.py: This script acts as the "Executive Brain". It is the user-facing part of the bot, featuring a real-time, rich-text user interface that displays trading statistics, recent trades, and the bot's status. Its primary functions include:

Connecting to the Deriv WebSocket API to get real-time price ticks.

Connecting to the local model.py WebSocket server to receive predictions.

Executing trades based on the model's predictions and managing risk with daily loss and profit limits.

Logging all activity and trade results to a file named main_bot.log.

model.py: This script is the "Deriv Terminator". It is the back-end "model engine" that performs the heavy lifting of data analysis and prediction. It runs as a separate process and is responsible for:

Collecting a buffer of tick data.

Generating a rich set of features from the buffered data, including statistical measures, streak analysis, frequency analysis, and various technical indicators like RSI and Bollinger Bands.

Training an ensemble of machine learning models, including RandomForestClassifier, GradientBoostingClassifier, MLPClassifier, and XGBClassifier.

Serving predictions to main.py via a WebSocket server running on localhost:8765.

Continuously retraining the models at a configured interval to adapt to changing market conditions.

How it Works
The main.py script launches model.py in the background. Once the model engine is initialized and a model_ready.flag file is created, main.py establishes a WebSocket connection to it. As new ticks arrive from the Deriv API, main.py forwards them to model.py for prediction. If the model's prediction confidence meets a set threshold, main.py places a trade on the Deriv platform.

2. System Architecture: main.py & model_trainer.py
This alternative, more robust architecture uses multiprocessing to run the analytical engine in a separate, dedicated process, using a SQLite database for data storage.

Core Components
main.py (main_rise_fall.py): As the "Final Blueprint", this script is the trading bot's entry point. Unlike the previous version, it uses the multiprocessing library to spawn a child process for the analytical engine. Key features include:

Connecting to the Deriv API and writing tick data directly to an SQLite database (tick_data.db).

Handling the trading logic and risk management.

Communicating with the child process to receive model predictions.

Using a dependency management function (install_and_import) to ensure all necessary libraries are installed automatically.

model_trainer.py: Described as the "Back-End Brain", this script runs independently in a separate process. Its job is to ingest data from the SQLite database and train sophisticated models. Its main responsibilities are:

Fetching clean data from the tick_data.db database.

Creating a rich feature set and labels from the tick data to train the models.

Training a variety of advanced models, including CatBoostClassifier, XGBClassifier, and a TCN neural network.

Saving the trained models to a models directory for the main bot to use for predictions.

Automatically installing required packages like websockets, numpy, and tensorflow to ensure a "flawless setup in any environment".

How it Works
The main.py script starts a separate process that runs model_trainer.py. As main.py receives ticks, it saves them to a shared SQLite database. The model_trainer.py process periodically checks this database, and once enough data is available, it trains new models and saves them. The main bot then uses these latest models to generate predictions for trading. This separation of concerns ensures that the data collection and model training do not block the real-time trading operations.
