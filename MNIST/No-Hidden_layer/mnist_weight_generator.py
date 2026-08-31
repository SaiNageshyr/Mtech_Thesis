'''import numpy as np
import os
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

print("1. Downloading MNIST dataset (this may take a minute)...")
# Download MNIST from OpenML
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

# X contains the 784 pixel values, y contains the labels (0-9)
X, y = mnist.data, mnist.target.astype(int)

# Normalize pixels to be between 0 and 1
X = X / 255.0

print("2. Training a simple Single-Layer Perceptron (Logistic Regression)...")
# We use Logistic Regression as it is mathematically identical to a single-layer perceptron
# solver='lbfgs' is fast, max_iter=100 is usually enough for a basic fit
model = LogisticRegression(solver='lbfgs', max_iter=500)
model.fit(X, y)

accuracy = model.score(X, y)
print(f"   -> Training Complete! Model Accuracy: {accuracy * 100:.2f}%")

print("3. Extracting and Quantizing Weights...")
# Extract the float weights and biases
# W is shape (10, 784). We transpose it to (784, 10) to match your MVM code.
W_float = model.coef_.T 
B_float = model.intercept_

# We will use 8-bit quantization for your hardware (-128 to +127)
# Find the maximum absolute weight to scale everything properly
max_val = max(np.max(np.abs(W_float)), np.max(np.abs(B_float)))
scale_factor = 127.0 / max_val

# Quantize and convert to integers
W_quantized = np.round(W_float * scale_factor).astype(int)
B_quantized = np.round(B_float * scale_factor).astype(int)

print("4. Saving files for the Hardware Simulator...")
# Create a directory to store the files
os.makedirs("mnist_data", exist_ok=True)

# Save the Weights matrix (784 rows, 10 columns)
np.savetxt("mnist_data/W_matrix.txt", W_quantized, fmt='%d')

# Save the Biases (10 values)
np.savetxt("mnist_data/B_vector.txt", B_quantized, fmt='%d')

# Save 5 sample images to test on your hardware
for i in range(5):
    sample_img = X[i]
    sample_label = y[i]
    # We also quantize the image pixels to 8-bit integers (0 to 127)
    sample_quantized = np.round(sample_img * 127).astype(int)
    
    np.savetxt(f"mnist_data/sample_img_{i}_label_{sample_label}.txt", sample_quantized, fmt='%d')
    print(f"   -> Saved Sample Image {i} (True Label: {sample_label})")

print("\nDone! You can now run your negative_MVM.py script.")'''

'''import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
from torch.utils.data import DataLoader, TensorDataset

# Set fixed seed for reproducible training
torch.manual_seed(42)
np.random.seed(42)

class FakeQuantizeSTE(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for Quantization-Aware Training.
    Forward Pass: Clamps and rounds floating-point values to simulate 8-bit integers.
    Backward Pass: Passes the gradient straight through unaltered (derivative = 1.0).
    """
    @staticmethod
    def forward(ctx, x, scale, min_val=-128, max_val=127):
        # 1. Scale continuous floats to integer space
        x_scaled = x / scale
        # 2. Clamp to 8-bit range and round
        x_clamped = torch.clamp(x_scaled, min_val, max_val)
        x_rounded = torch.round(x_clamped)
        # 3. De-quantize back to float scale for forward propagation
        return x_rounded * scale

    @staticmethod
    def backward(ctx, grad_output):
        # STE Trick: Pretend derivative of round() is 1.0
        return grad_output, None, None, None

class QAT_SingleLayerPerceptron(nn.Module):
    def __init__(self, in_features=784, out_features=10):
        super(QAT_SingleLayerPerceptron, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        # Dynamically compute scale factors for 8-bit representation (-128 to 127)
        w_max = torch.max(torch.abs(self.fc.weight))
        w_scale = torch.clamp(w_max / 127.0, min=1e-8)

        x_max = torch.max(torch.abs(x))
        x_scale = torch.clamp(x_max / 127.0, min=1e-8)

        # Inject fake quantization noise during training
        w_qat = FakeQuantizeSTE.apply(self.fc.weight, w_scale, -128, 127)
        x_qat = FakeQuantizeSTE.apply(x, x_scale, 0, 127)

        # Compute output with fake-quantized parameters
        out = torch.matmul(x_qat, w_qat.T) + self.fc.bias
        return out

print("1. Downloading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_raw, y_raw = mnist.data, mnist.target.astype(int)

# Normalize inputs between 0.0 and 1.0
X_norm = (X_raw / 255.0).astype(np.float32)

# Convert to PyTorch Tensors
X_tensor = torch.tensor(X_norm)
y_tensor = torch.tensor(y_raw, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

print("2. Training Single-Layer Perceptron using Quantization-Aware Training (QAT)...")
model = QAT_SingleLayerPerceptron()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

epochs = 5
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_acc = (correct / total) * 100
    print(f"   -> Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/total:.4f} | QAT Training Accuracy: {epoch_acc:.2f}%")

print("\n3. Extracting True INT8 Integer Matrices for Crossbar Hardware...")
model.eval()

with torch.no_grad():
    W_float = model.fc.weight.numpy().T  # Transpose to shape (784, 10)
    B_float = model.fc.bias.numpy()

    # Exact scale calculation matching QAT training
    max_val = max(np.max(np.abs(W_float)), np.max(np.abs(B_float)))
    scale_factor = 127.0 / max_val

    # Convert to pure signed 8-bit integers
    W_quantized = np.round(W_float * scale_factor).astype(int)
    B_quantized = np.round(B_float * scale_factor).astype(int)

print("4. Saving integer files to 'mnist_data/' directory...")
os.makedirs("mnist_data", exist_ok=True)

np.savetxt("mnist_data/W_matrix.txt", W_quantized, fmt='%d')
np.savetxt("mnist_data/B_vector.txt", B_quantized, fmt='%d')

# Save 5 sample images quantized to 8-bit integer range (0 to 127)
for i in range(5):
    sample_img = X_norm[i]
    sample_label = y_raw[i]
    sample_quantized = np.round(sample_img * 127).astype(int)
    
    np.savetxt(f"mnist_data/sample_img_{i}_label_{sample_label}.txt", sample_quantized, fmt='%d')
    print(f"   -> Saved Sample Image {i} (True Label: {sample_label})")

print("\nDone! QAT-trained integer weight files generated successfully.")'''


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
from torch.utils.data import DataLoader, TensorDataset

# Set fixed seed for reproducible training
torch.manual_seed(42)
np.random.seed(42)

class FakeQuantizeSTE(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for Quantization-Aware Training.
    Forward Pass: Clamps and rounds floating-point values to simulate 8-bit integers.
    Backward Pass: Passes the gradient straight through unaltered (derivative = 1.0).
    """
    @staticmethod
    def forward(ctx, x, scale, min_val=-128, max_val=127):
        x_scaled = x / scale
        x_clamped = torch.clamp(x_scaled, min_val, max_val)
        x_rounded = torch.round(x_clamped)
        return x_rounded * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None

class QAT_SingleLayerPerceptron(nn.Module):
    def __init__(self, in_features=784, out_features=10):
        super(QAT_SingleLayerPerceptron, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        w_max = torch.max(torch.abs(self.fc.weight))
        w_scale = torch.clamp(w_max / 127.0, min=1e-8)

        x_max = torch.max(torch.abs(x))
        x_scale = torch.clamp(x_max / 127.0, min=1e-8)

        w_qat = FakeQuantizeSTE.apply(self.fc.weight, w_scale, -128, 127)
        x_qat = FakeQuantizeSTE.apply(x, x_scale, 0, 127)

        out = torch.matmul(x_qat, w_qat.T) + self.fc.bias
        return out

print("1. Downloading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_raw, y_raw = mnist.data, mnist.target.astype(int)

X_norm = (X_raw / 255.0).astype(np.float32)
X_tensor = torch.tensor(X_norm)
y_tensor = torch.tensor(y_raw, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

print("2. Training Single-Layer Perceptron using Quantization-Aware Training (QAT)...")
model = QAT_SingleLayerPerceptron()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

epochs = 5
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_acc = (correct / total) * 100
    print(f"   -> Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/total:.4f} | QAT Training Accuracy: {epoch_acc:.2f}%")

print("\n3. Extracting Float32 and INT8 Parameters for Crossbar Hardware...")
model.eval()

with torch.no_grad():
    W_float = model.fc.weight.numpy().T  # Shape: (784, 10)
    B_float = model.fc.bias.numpy()

    max_val = max(np.max(np.abs(W_float)), np.max(np.abs(B_float)))
    scale_factor = 127.0 / max_val

    W_quantized = np.round(W_float * scale_factor).astype(int)
    B_quantized = np.round(B_float * scale_factor).astype(int)

print("4. Saving parameter files to 'mnist_data/' directory...")
os.makedirs("mnist_data", exist_ok=True)

np.savetxt("mnist_data/W_matrix.txt", W_quantized, fmt='%d')
np.savetxt("mnist_data/B_vector.txt", B_quantized, fmt='%d')
np.savetxt("mnist_data/W_float.txt", W_float, fmt='%.6f')
np.savetxt("mnist_data/B_float.txt", B_float, fmt='%.6f')

for i in range(5):
    sample_img = X_norm[i]
    sample_label = y_raw[i]
    sample_quantized = np.round(sample_img * 127).astype(int)
    
    np.savetxt(f"mnist_data/sample_img_{i}_label_{sample_label}.txt", sample_quantized, fmt='%d')
    np.savetxt(f"mnist_data/sample_img_{i}_float.txt", sample_img, fmt='%.6f')
    print(f"   -> Saved Sample Image {i} (True Label: {sample_label})")

print("\nDone! QAT-trained integer and float weight files generated successfully.")
