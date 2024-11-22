from prometheus_client import start_http_server, Gauge, CollectorRegistry, REGISTRY 

from sklearn.ensemble import RandomForestClassifier 

from sklearn.datasets import load_iris 

from sklearn.model_selection import train_test_split 

from sklearn.metrics import accuracy_score 

import random 

import time 

 

# Clear previous Prometheus metrics to avoid duplication issues 

registry = CollectorRegistry() 

for collector in list(REGISTRY._collector_to_names.keys()): 

    REGISTRY.unregister(collector) 

 

# Initialize Prometheus metrics 

accuracy_metric = Gauge('model_accuracy', 'Model accuracy', registry=registry) 

prediction_metric = Gauge('model_predictions', 'Model prediction count', registry=registry) 

error_metric_v2 = Gauge('model_errors_v2', 'Model error count', registry=registry)  # Renamed to avoid duplicate name error 

 

# Load dataset and split it 

data = load_iris() 

X = data.data 

y = data.target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

 

# Train a simple model 

model = RandomForestClassifier(n_estimators=10, random_state=42) 

model.fit(X_train, y_train) 

 

 

# Start Prometheus HTTP server on port 8000 

start_http_server(8000) 

 

# Function to simulate monitoring and logging 

def monitor_model(): 

    # Make predictions and calculate accuracy 

    predictions = model.predict(X_test) 

    accuracy = accuracy_score(y_test, predictions) 

 

    # Update Prometheus metrics 

    accuracy_metric.set(accuracy)                  # Log model accuracy 

    prediction_metric.set(len(predictions))        # Log number of predictions made 

    # Randomly simulate errors for demonstration 

    simulated_errors = random.randint(0, 5) 

    error_metric_v2.set(simulated_errors)          # Log number of simulated errors 

 

    print(f"Model Accuracy: {accuracy}") 

    print(f"Total Predictions: {len(predictions)}") 

    print(f"Simulated Errors: {simulated_errors}") 

 

# Periodically update metrics 

while True: 

    monitor_model() 

    time.sleep(10)  # Sleep for 10 seconds before logging again 

 