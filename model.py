# model.py
import asyncio
import websockets
import json
import numpy as np
import time
import pandas as pd
from datetime import datetime, timedelta
from collections import deque, Counter, defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import uniform, randint
import joblib
import os
import warnings
from scipy import stats, signal
from typing import Dict, List, Deque, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import logging

warnings.filterwarnings('ignore')

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("model_log.log"),
                        logging.StreamHandler()
                    ])

# --- CONFIG ---
SYMBOL = "1HZ10V"
TICK_BUFFER_SIZE = 7200  # Amount of ticks stored for retraining
MODEL_UPDATE_INTERVAL_SECONDS = 7200 # Time-based update interval
MIN_TRAINING_SAMPLES = 500
MODEL_COMM_PORT = 8765
MODEL_READY_FILE = "model_ready.flag"
MODEL_DIR = "models"

# --- NEW CONFIG FOR REVAMP ---
MIN_ACCEPTABLE_ACCURACY = 0.75 # If ensemble accuracy drops below this, trigger retraining

class DerivTerminator:
    def __init__(self):
        self.last_digits_buffer: Deque[int] = deque(maxlen=TICK_BUFFER_SIZE)
        self.last_tick_times: Deque[int] = deque(maxlen=TICK_BUFFER_SIZE)
        self.active_models: Dict[str, any] = {}
        self.scaler = PowerTransformer(method='yeo-johnson')
        self.last_prediction: int = -1
        self.prediction_confidence: float = 0.0
        self.last_training_time: float = 0.0
        self.model_performance: List[Dict] = [] # Tracks individual model accuracies per training run
        self.current_ensemble_accuracy: float = 0.0 # Stores the latest ensemble accuracy
        self.current_prediction_probabilities: Dict[int, float] = {} # Stores the latest probabilities
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.prediction_event = asyncio.Event() # Event to signal new prediction is ready

terminator = DerivTerminator()

def get_last_digit_from_tick(tick_data: Dict) -> int:
    price = float(tick_data['tick']['quote'])
    return int(str(price).split('.')[-1][-1])

# Calculate RSI
def calculate_rsi(series, window):
    diff = series.diff().dropna()
    gain = diff.mask(diff < 0, 0)
    loss = diff.mask(diff > 0, 0).abs()
    avg_gain = gain.ewm(span=window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(span=window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate Bollinger Bands
def calculate_bollinger_bands(series, window, num_std_dev):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + (std * num_std_dev)
    lower_band = sma - (std * num_std_dev)
    return sma, upper_band, lower_band

# Generate complex pattern features from digit and time buffers
def generate_pattern_features(digits_buffer: Deque[int], tick_times_buffer: Deque[int]) -> pd.DataFrame:
    min_required_samples = 50
    if len(digits_buffer) < min_required_samples:
        logging.debug(f"Not enough data for feature generation: {len(digits_buffer)} < {min_required_samples}")
        return pd.DataFrame()

    df = pd.DataFrame({'digit': list(digits_buffer), 'time': list(tick_times_buffer)})
    df['time'] = pd.to_datetime(df['time'], unit='s')

    features = {}

    # Basic Statistical Features
    features['std_digits'] = np.std(df['digit'])
    features['mean_digits'] = np.mean(df['digit'])
    features['median_digits'] = np.median(df['digit'])
    features['skew_digits'] = stats.skew(df['digit'])
    features['kurtosis_digits'] = stats.kurtosis(df['digit'])
    features['max_digit'] = np.max(df['digit'])
    features['min_digit'] = np.min(df['digit'])
    features['range_digits'] = features['max_digit'] - features['min_digit']

    # Streak Analysis
    current_lower, current_upper = 0, 0
    lower_streaks, upper_streaks = [], []
    for digit in df['digit']:
        if 0 <= digit <= 2:
            current_lower += 1
            if current_upper > 0: upper_streaks.append(current_upper)
            current_upper = 0
        elif 7 <= digit <= 9:
            current_upper += 1
            if current_lower > 0: lower_streaks.append(current_lower)
            current_lower = 0
        else:
            if current_lower > 0: lower_streaks.append(current_lower)
            if current_upper > 0: upper_streaks.append(current_upper)
            current_lower, current_upper = 0, 0
    if current_lower > 0: lower_streaks.append(current_lower)
    if current_upper > 0: upper_streaks.append(current_upper)

    features['current_lower_streak'] = current_lower
    features['current_upper_streak'] = current_upper
    features['avg_lower_streak_len'] = np.mean(lower_streaks) if lower_streaks else 0
    features['max_lower_streak_len'] = np.max(lower_streaks) if lower_streaks else 0
    features['avg_upper_streak_len'] = np.mean(upper_streaks) if upper_streaks else 0
    features['max_upper_streak_len'] = np.max(upper_streaks) if upper_streaks else 0

    # Frequency Analysis
    digit_counts_series = df['digit'].value_counts(normalize=True).reindex(range(10), fill_value=0)
    features['most_common_digit_val'] = digit_counts_series.idxmax()
    features['least_common_digit_val'] = digit_counts_series.idxmin()
    features['entropy_last_digits'] = stats.entropy(digit_counts_series.values)
    for i in range(10):
        features[f'freq_digit_{i}'] = digit_counts_series.get(i, 0)

    # Time-based Features
    time_diffs = df['time'].diff().dt.total_seconds().dropna()
    features['avg_time_between_ticks'] = time_diffs.mean() if not time_diffs.empty else 0
    features['std_time_between_ticks'] = time_diffs.std() if len(time_diffs) > 1 else 0
    features['max_time_between_ticks'] = time_diffs.max() if not time_diffs.empty else 0
    features['min_time_between_ticks'] = time_diffs.min() if not time_diffs.empty else 0

    # Lagged Features
    for i in range(1, 11): # Lagged digits up to 10 ticks back
        features[f'lag_digit_{i}'] = digits_buffer[-1 - i] if len(digits_buffer) > i else 0

    # Moving Averages for Digits
    windows = [5, 10, 20, 50]
    for w in windows:
        if len(df['digit']) >= w:
            features[f'sma_digit_{w}'] = df['digit'].rolling(window=w).mean().iloc[-1]
            features[f'ema_digit_{w}'] = df['digit'].ewm(span=w, adjust=False).mean().iloc[-1]
        else:
            features[f'sma_digit_{w}'] = features[f'ema_digit_{w}'] = features['mean_digits']

    # Relative Strength Index (RSI) for Digits
    if len(df['digit']) >= 14:
        rsi_val = calculate_rsi(df['digit'], 14).iloc[-1]
        features['rsi_digit_14'] = rsi_val if not pd.isna(rsi_val) else 50
    else:
        features['rsi_digit_14'] = 50

    # Bollinger Bands for Digits (Deviation from SMA)
    if len(df['digit']) >= 20:
        sma_bb, upper_bb, lower_bb = calculate_bollinger_bands(df['digit'], 20, 2)
        features['bb_dist_upper'] = df['digit'].iloc[-1] - upper_bb.iloc[-1] if not pd.isna(upper_bb.iloc[-1]) else 0
        features['bb_dist_lower'] = df['digit'].iloc[-1] - lower_bb.iloc[-1] if not pd.isna(lower_bb.iloc[-1]) else 0
        features['bb_width'] = upper_bb.iloc[-1] - lower_bb.iloc[-1] if not pd.isna(upper_bb.iloc[-1]) and not pd.isna(lower_bb.iloc[-1]) else 0
    else:
        features['bb_dist_upper'] = features['bb_dist_lower'] = features['bb_width'] = 0

    # Difference from most common digit
    features['diff_from_most_common'] = df['digit'].iloc[-1] - features['most_common_digit_val']

    # Digit change momentum
    df['digit_diff'] = df['digit'].diff().fillna(0)
    features['digit_change_momentum_5'] = df['digit_diff'].rolling(window=5).sum().iloc[-1] if len(df['digit_diff']) >= 5 else 0
    features['digit_change_momentum_10'] = df['digit_diff'].rolling(window=10).sum().iloc[-1] if len(df['digit_diff']) >= 10 else 0

    return pd.DataFrame([features])

async def train_terminator_model():
    if len(terminator.last_digits_buffer) < MIN_TRAINING_SAMPLES:
        logging.info(f"Skipping training: insufficient buffer size ({len(terminator.last_digits_buffer)} < {MIN_TRAINING_SAMPLES})")
        return

    logging.info(f"Training models with {len(terminator.last_digits_buffer)} ticks available.")

    full_data_df = pd.DataFrame({
        'digit': list(terminator.last_digits_buffer),
        'time': list(terminator.last_tick_times)
    })

    training_window_size = 150
    
    all_features = []
    targets = []

    feature_generation_tasks = []
    for i in range(len(full_data_df) - training_window_size):
        window_digits_data = full_data_df['digit'].iloc[i : i + training_window_size]
        window_times_data = full_data_df['time'].iloc[i : i + training_window_size]
        feature_generation_tasks.append(
            asyncio.get_event_loop().run_in_executor(
                terminator.executor, generate_pattern_features, deque(window_digits_data.tolist()), deque(window_times_data.tolist())
            )
        )

    features_dfs = await asyncio.gather(*feature_generation_tasks)

    for i, features_df in enumerate(features_dfs):
        if not features_df.empty:
            target_idx = i + training_window_size
            if target_idx < len(full_data_df):
                all_features.append(features_df.iloc[0].to_dict())
                targets.append(full_data_df['digit'].iloc[target_idx])

    if not all_features:
        logging.warning("Not enough training data points after feature generation.")
        return

    training_df = pd.DataFrame(all_features).dropna()
    y_train = pd.Series(targets).iloc[training_df.index].astype(int)

    if training_df.empty or y_train.empty:
        logging.warning("Training DataFrame or targets are empty after dropping NaNs.")
        return

    X_train = training_df

    sample_features_check = generate_pattern_features(deque(range(min(training_window_size, TICK_BUFFER_SIZE))), deque(range(min(training_window_size, TICK_BUFFER_SIZE), min(training_window_size, TICK_BUFFER_SIZE)*2)))
    if not sample_features_check.empty:
        all_feature_names = sample_features_check.columns.tolist()
        X_train = X_train.reindex(columns=all_feature_names, fill_value=0)
    else:
        logging.error("Failed to generate sample features for reindexing. Skipping training.")
        return

    try:
        terminator.scaler.fit(X_train)
        X_scaled = terminator.scaler.transform(X_train)
    except ValueError as e:
        logging.error(f"SCALER FITTING ERROR: {e}")
        return

    models_to_train_config = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': randint(100, 500),
                'max_depth': randint(10, 30),
                'min_samples_leaf': randint(1, 10),
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': randint(100, 300),
                'learning_rate': uniform(0.01, 0.1),
                'max_depth': randint(3, 10),
            }
        },
        'MLPClassifier': {
            'model': MLPClassifier(random_state=42, early_stopping=True, n_iter_no_change=20),
            'params': {
                'hidden_layer_sizes': [(randint(50, 200).rvs(), randint(30, 100).rvs(), randint(10, 50).rvs())],
                'activation': ['relu', 'tanh'],
                'solver': ['adam'],
                'max_iter': randint(300, 600),
            }
        },
        'XGBoost': {
            'model': XGBClassifier(objective='multi:softmax', num_class=10, use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': randint(100, 400),
                'learning_rate': uniform(0.01, 0.15),
                'max_depth': randint(3, 12),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
            }
        }
    }

    current_accuracies = {}
    tscv = TimeSeriesSplit(n_splits=5)

    for name, config in models_to_train_config.items():
        logging.info(f"Hyperparameter tuning and training {name}...")
        try:
            rand_search = RandomizedSearchCV(
                estimator=config['model'],
                param_distributions=config['params'],
                n_iter=20,
                cv=tscv,
                scoring='accuracy',
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            rand_search.fit(X_scaled, y_train)

            best_model = rand_search.best_estimator_
            terminator.active_models[name] = best_model
            joblib.dump(best_model, os.path.join(MODEL_DIR, f'{name}_model.joblib'))

            y_pred = best_model.predict(X_scaled)
            accuracy = accuracy_score(y_train, y_pred)
            current_accuracies[name] = accuracy
            logging.info(f"Accuracy for {name} (Best Params: {rand_search.best_params_}): {accuracy:.2%}")
        except Exception as e:
            logging.error(f"Error training {name}: {e}")
            if name in terminator.active_models: del terminator.active_models[name]

    joblib.dump(terminator.scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
    logging.info(f"Models trained on {len(X_train)} samples.")

    if current_accuracies:
        terminator.model_performance.append({
            'timestamp': datetime.now(),
            'total_samples': len(X_train),
            **current_accuracies
        })
        latest_accuracies = [acc for model_name, acc in current_accuracies.items() if model_name in terminator.active_models]
        terminator.current_ensemble_accuracy = np.mean(latest_accuracies) if latest_accuracies else 0.0

    terminator.last_training_time = time.time()

def calculate_current_prediction():
    # This function is now synchronous (or uses executor) and called by deriv_websocket_handler
    min_prediction_samples = 50
    if not terminator.active_models or len(terminator.last_digits_buffer) < min_prediction_samples:
        logging.warning("Prediction skipped: Model not ready or insufficient data.")
        terminator.last_prediction = -1
        terminator.prediction_confidence = 0.0
        terminator.current_prediction_probabilities = {}
        terminator.prediction_event.clear()
        return

    try:
        # Running feature generation in the thread pool to avoid blocking the main event loop
        features_df = terminator.executor.submit(generate_pattern_features, terminator.last_digits_buffer, terminator.last_tick_times).result()

        if features_df.empty:
            logging.warning("Prediction skipped: Feature generation failed.")
            terminator.last_prediction = -1
            terminator.prediction_confidence = 0.0
            terminator.current_prediction_probabilities = {}
            terminator.prediction_event.clear()
            return

        if not hasattr(terminator.scaler, 'n_features_in_'):
            logging.warning("Prediction skipped: Scaler not fitted.")
            terminator.last_prediction = -1
            terminator.prediction_confidence = 0.0
            terminator.current_prediction_probabilities = {}
            terminator.prediction_event.clear()
            return

        expected_features = terminator.scaler.feature_names_in_
        features_df = features_df.reindex(columns=expected_features, fill_value=0)

        X_pred_scaled = terminator.scaler.transform(features_df)

        ensemble_probabilities: Dict[int, float] = defaultdict(float)
        num_active_models = 0

        for name, model in terminator.active_models.items():
            try:
                proba = model.predict_proba(X_pred_scaled)[0]
                for digit_idx in range(10):
                    ensemble_probabilities[digit_idx] += proba[digit_idx]
                num_active_models += 1
            except Exception as e:
                logging.error(f"Error in {name} prediction: {e}")

        if num_active_models == 0:
            logging.warning("No successful predictions from active models for ensemble.")
            terminator.last_prediction = -1
            terminator.prediction_confidence = 0.0
            terminator.current_prediction_probabilities = {}
            terminator.prediction_event.clear()
            return

        for digit in ensemble_probabilities:
            ensemble_probabilities[digit] /= num_active_models

        if ensemble_probabilities:
            final_probabilities = dict(ensemble_probabilities)
            best_digit = max(final_probabilities, key=final_probabilities.get)
            confidence_of_best_digit = final_probabilities[best_digit]
        else:
            final_probabilities = {}
            best_digit = -1
            confidence_of_best_digit = 0.0

        terminator.last_prediction = int(best_digit)
        terminator.prediction_confidence = float(confidence_of_best_digit)
        terminator.current_prediction_probabilities = final_probabilities
        terminator.prediction_event.set() # Signal that a new prediction is ready

    except Exception as e:
        logging.error(f"PREDICTION ERROR: {e}")
        terminator.last_prediction = -1
        terminator.prediction_confidence = 0.0
        terminator.current_prediction_probabilities = {}
        terminator.prediction_event.clear()


async def deriv_websocket_handler():
    uri = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                logging.info(f"Connected to Deriv WebSocket for symbol {SYMBOL}.")
                await websocket.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))
                async for message in websocket:
                    data = json.loads(message)
                    if data.get("msg_type") == "tick":
                        last_digit = get_last_digit_from_tick(data)
                        epoch_time = int(data['tick']['epoch'])
                        terminator.last_digits_buffer.append(last_digit)
                        terminator.last_tick_times.append(epoch_time)

                        # Immediately calculate prediction after a new tick
                        # Use run_in_executor for CPU-bound prediction to not block WS handler
                        await asyncio.get_event_loop().run_in_executor(
                            terminator.executor, calculate_current_prediction
                        )
                        terminator.prediction_event.set() # Ensure event is set after prediction calculation

                        # Trigger training based on time interval or accuracy degradation
                        time_for_update = (time.time() - terminator.last_training_time) >= MODEL_UPDATE_INTERVAL_SECONDS
                        accuracy_low = terminator.current_ensemble_accuracy < MIN_ACCEPTABLE_ACCURACY and terminator.current_ensemble_accuracy != 0.0

                        if (time_for_update or accuracy_low) and \
                           len(terminator.last_digits_buffer) >= MIN_TRAINING_SAMPLES:
                            logging.info(f"Initiating training. Time for update: {time_for_update}, Accuracy low: {accuracy_low}. Buffer: {len(terminator.last_digits_buffer)}.")
                            await train_terminator_model()
        except Exception as e:
            logging.error(f"Deriv WS ERROR: {e} - Reconnecting in 3s")
            await asyncio.sleep(3)

async def prediction_server_handler(websocket, path):
    try:
        async for message in websocket:
            data = json.loads(message)
            if data.get("request") == "prediction":
                # Wait for a new prediction to be ready (signaled by the event)
                # This ensures main.py gets the LATEST prediction after a tick
                await terminator.prediction_event.wait()
                terminator.prediction_event.clear() # Clear the event for the next prediction

                # Return the pre-calculated prediction
                response = {
                    "probabilities": terminator.current_prediction_probabilities,
                    "prediction": terminator.last_prediction,
                    "confidence": terminator.prediction_confidence,
                    "accuracy": terminator.current_ensemble_accuracy,
                    "barrier_over": 2, # Fixed
                    "barrier_under": 7 # Fixed
                }
                await websocket.send(json.dumps(response))
            else:
                await websocket.send(json.dumps({"error": "Invalid request"}))
    except websockets.exceptions.ConnectionClosedOK:
        logging.info("Prediction server client disconnected.")
    except Exception as e:
        logging.error(f"PREDICTION SERVER HANDLER ERROR: {e}")

async def start_prediction_server():
    server = await websockets.serve(prediction_server_handler, "localhost", MODEL_COMM_PORT)
    logging.info(f"Prediction server listening on ws://localhost:{MODEL_COMM_PORT}")
    await server.wait_closed()

async def terminator_loop():
    asyncio.create_task(deriv_websocket_handler())
    asyncio.create_task(start_prediction_server())

    if os.path.exists(os.path.join(MODEL_DIR, 'scaler.joblib')):
        try:
            terminator.scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
            logging.info("Scaler loaded.")
        except Exception as e:
            logging.error(f"Error loading scaler: {e}")

    for model_name in ['RandomForest', 'GradientBoosting', 'MLPClassifier', 'XGBoost']:
        model_path = os.path.join(MODEL_DIR, f'{model_name}_model.joblib')
        if os.path.exists(model_path):
            try:
                terminator.active_models[model_name] = joblib.load(model_path)
                logging.info(f"{model_name} loaded.")
            except Exception as e:
                logging.error(f"Error loading {model_name}: {e}")

    if terminator.active_models and terminator.model_performance:
        if terminator.model_performance:
            latest_accuracies_dict = terminator.model_performance[-1]
            active_model_accs = [latest_accuracies_dict[name] for name in terminator.active_models if name in latest_accuracies_dict]
            terminator.current_ensemble_accuracy = np.mean(active_model_accs) if active_model_accs else 0.0
            logging.info(f"Initial ensemble accuracy: {terminator.current_ensemble_accuracy:.2%}")
        else:
            logging.info("No past model performance data to calculate initial ensemble accuracy.")

    with open(MODEL_READY_FILE, "w") as f:
        f.write("ready")
    logging.info(f"Model ready signal sent via {MODEL_READY_FILE}")

    last_print_time = time.time()
    while True:
        if time.time() - last_print_time >= 1:
            last_print_time = time.time()
            current_buffer_size = len(terminator.last_digits_buffer)
            model_status = "Ready" if terminator.active_models else "Training..."

            logging.info(f"Buffer: {current_buffer_size}/{TICK_BUFFER_SIZE} | "
                         f"Models: {model_status} | "
                         f"Last Pred: {terminator.last_prediction} (Conf: {terminator.prediction_confidence:.2%}) | "
                         f"Ensemble Acc: {terminator.current_ensemble_accuracy:.2%}")
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    logging.info("Starting Deriv Terminator - Revamped")
    try:
        asyncio.run(terminator_loop())
    except KeyboardInterrupt:
        logging.info("Shutting down - Revamped")
    finally:
        if os.path.exists(MODEL_READY_FILE):
            os.remove(MODEL_READY_FILE)