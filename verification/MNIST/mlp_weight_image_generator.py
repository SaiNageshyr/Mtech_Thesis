import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import fetch_openml

# -----------------------------------------------------------------------------
# 1. Straight-Through Estimator (STE) Fake Quantization Function
# -----------------------------------------------------------------------------
class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, min_val, max_val):
        scaled = input * scale
        rounded = torch.round(scaled)
        clamped = torch.clamp(rounded, min_val, max_val)
        # De-quantize back to float domain for fake-quantization
        return clamped / scale

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: pass gradients unchanged
        return grad_output, None, None, None

def fake_quantize(input, scale, min_val, max_val):
    return STEQuantize.apply(input, scale, min_val, max_val)

# -----------------------------------------------------------------------------
# 2. QAT Neural Network Architecture
# -----------------------------------------------------------------------------
class QAT_MLP(nn.Module):
    def __init__(self, target_bits):
        super(QAT_MLP, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)
        
        self.target_bits = target_bits
        self.max_signed_val = float(min((1 << (target_bits - 1)) - 1, 10000))

    def fake_quantize_tensor(self, tensor):
        max_val = torch.max(torch.abs(tensor))
        if max_val == 0:
            return tensor
        scale = self.max_signed_val / max_val
        return fake_quantize(tensor, scale, -self.max_signed_val, self.max_signed_val)

    def forward(self, x):
        # Apply STE Fake-Quantization to weights and biases during forward pass
        w1_q = self.fake_quantize_tensor(self.fc1.weight)
        b1_q = self.fake_quantize_tensor(self.fc1.bias)
        
        x = nn.functional.linear(x, w1_q, b1_q)
        x = torch.relu(x)

        w2_q = self.fake_quantize_tensor(self.fc2.weight)
        b2_q = self.fake_quantize_tensor(self.fc2.bias)

        x = nn.functional.linear(x, w2_q, b2_q)
        return x

    def extract_quantized_weights(self):
        """Extract final PyTorch tensors as quantized hardware integers and floats."""
        w1_f = self.fc1.weight.detach().cpu().numpy().T  # Shape: (784, 64)
        b1_f = self.fc1.bias.detach().cpu().numpy()      # Shape: (64,)
        w2_f = self.fc2.weight.detach().cpu().numpy().T  # Shape: (64, 10)
        b2_f = self.fc2.bias.detach().cpu().numpy()      # Shape: (10,)

        scale1 = self.max_signed_val / max(np.max(np.abs(w1_f)), np.max(np.abs(b1_f)))
        scale2 = self.max_signed_val / max(np.max(np.abs(w2_f)), np.max(np.abs(b2_f)))

        w1_q = np.round(w1_f * scale1).astype(np.int64)
        b1_q = np.round(b1_f * scale1).astype(np.int64)
        w2_q = np.round(w2_f * scale2).astype(np.int64)
        b2_q = np.round(b2_f * scale2).astype(np.int64)

        return (w1_q, b1_q, w2_q, b2_q), (w1_f, b1_f, w2_f, b2_f)

# -----------------------------------------------------------------------------
# 3. Main Execution Block
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 mlp_weight_image_generator.py <bit_width>")
        sys.exit(1)

    TARGET_BITS = int(sys.argv[1])
    MAX_SIGNED_VAL = min((1 << (TARGET_BITS - 1)) - 1, 10000)
    MAX_PIXEL_VAL = min((1 << TARGET_BITS) - 1, 255)
    
    DATA_DIR = f"mnist_mlp_data_{TARGET_BITS}"

    print("1. Downloading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data.astype(np.float32), mnist.target.astype(np.int64)
    X /= 255.0  # Normalize pixels [0.0, 1.0]

    # Convert to PyTorch Tensors
    tensor_x = torch.tensor(X)
    tensor_y = torch.tensor(y)
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    print(f"2. QAT Training 1-Hidden-Layer MLP (784 -> 64 -> 10) for {TARGET_BITS}-bit...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QAT_MLP(target_bits=TARGET_BITS).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    # QAT Training Loop
    epochs = 10
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()  # STE passes gradients through fake-quantization
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"   Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(dataloader):.4f}")

    # Evaluate Accuracy
    model.eval()
    with torch.no_grad():
        test_outputs = model(tensor_x.to(device))
        predictions = torch.argmax(test_outputs, dim=1).cpu().numpy()
        accuracy = np.mean(predictions == y)
    print(f"   -> QAT Training Complete! Model Accuracy: {accuracy * 100:.2f}%")

    print(f"3. Extracting and Saving QAT Quantized Weights for {TARGET_BITS}-bit Hardware...")
    (W1_quant, B1_quant, W2_quant, B2_quant), (W1_float, B1_float, W2_float, B2_float) = model.extract_quantized_weights()

    os.makedirs(DATA_DIR, exist_ok=True)

    # Save quantized integer files
    np.savetxt(f"{DATA_DIR}/W1_matrix.txt", W1_quant, fmt='%d')
    np.savetxt(f"{DATA_DIR}/B1_vector.txt", B1_quant, fmt='%d')
    np.savetxt(f"{DATA_DIR}/W2_matrix.txt", W2_quant, fmt='%d')
    np.savetxt(f"{DATA_DIR}/B2_vector.txt", B2_quant, fmt='%d')

    # Save original float files
    np.savetxt(f"{DATA_DIR}/W1_float.txt", W1_float, fmt='%.6f')
    np.savetxt(f"{DATA_DIR}/B1_float.txt", B1_float, fmt='%.6f')
    np.savetxt(f"{DATA_DIR}/W2_float.txt", W2_float, fmt='%.6f')
    np.savetxt(f"{DATA_DIR}/B2_float.txt", B2_float, fmt='%.6f')

    print(f"4. Generating 1000 Sample Images in '{DATA_DIR}/'...")
    for i in range(1000):
        sample_img = X[i]
        sample_label = y[i]
        sample_quantized = np.round(sample_img * MAX_PIXEL_VAL).astype(int)
        
        np.savetxt(f"{DATA_DIR}/sample_img_{i}_label_{sample_label}.txt", sample_quantized, fmt='%d')
        np.savetxt(f"{DATA_DIR}/sample_img_{i}_float.txt", sample_img, fmt='%.6f')

    print(f"\nDone! All {TARGET_BITS}-bit QAT trained MLP files generated successfully.")
