import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from scapy.all import IP, ICMP, TCP, send, sniff

# Function to generate normal network traffic
def generate_normal_traffic(destination_ip, num_packets=500):
    for _ in range(num_packets):
        packet = IP(dst=destination_ip) / TCP()
        send(packet, verbose=0)

# Function to simulate a network issue anomaly
def simulate_network_issue(destination_ip, num_anomalous_packets=50):
    for _ in range(num_anomalous_packets):
        packet = IP(dst=destination_ip) / ICMP()  # Simulating an ICMP anomaly
        send(packet, verbose=0)

# Generate normal network traffic
normal_destination_ip = "192.168.1.1"  
generate_normal_traffic(normal_destination_ip)

# Simulate a network issue anomaly
anomalous_destination_ip = "192.168.1.2"  
simulate_network_issue(anomalous_destination_ip)

# Combine normal and anomalous data
normal_data = np.random.normal(0, 1, (500, 2))  # Normal data
anomalous_data = np.random.uniform(4, 6, (50, 2))  # Anomalous data

data = np.vstack([normal_data, anomalous_data])
labels = np.hstack([np.zeros(len(normal_data)), np.ones(len(anomalous_data))])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train the Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Convert predictions to 0 (normal) or 1 (anomalous)
y_pred[y_pred == 1] = 0  # 1 represents normal, 0 represents anomalous
y_pred[y_pred == -1] = 1

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualize the results
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', marker='o', label='Actual')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', marker='x', s=100, label='Predicted Anomalies')
plt.legend()
plt.title('Anomaly Detection Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
