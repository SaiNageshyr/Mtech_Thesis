import numpy as np
import os
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

print("1. Downloading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data, mnist.target.astype(int)

# Normalize input pixels between 0.0 and 1.0
X = X / 255.0

print("2. Training 1-Hidden-Layer MLP (784 -> 64 -> 10)...")
model = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam', max_iter=20, random_state=42)
model.fit(X, y)

accuracy = model.score(X, y)
print(f"   -> Training Complete! Model Accuracy: {accuracy * 100:.2f}%")

print("3. Extracting and Safely Quantizing Weights for 32-bit Hardware...")
W1_float = model.coefs_[0]      # Shape: (784, 64)
B1_float = model.intercepts_[0] # Shape: (64,)
W2_float = model.coefs_[1]      # Shape: (64, 10)
B2_float = model.intercepts_[1] # Shape: (10,)

# Scale weights so Layer 1 accumulation (784 terms x max_pixel_255 x max_weight_1000)
# strictly stays under 2,147,483,647 (signed 32-bit max integer limit).
max_val1 = max(np.max(np.abs(W1_float)), np.max(np.abs(B1_float)))
scale_factor1 = 1000.0 / max_val1

max_val2 = max(np.max(np.abs(W2_float)), np.max(np.abs(B2_float)))
scale_factor2 = 1000.0 / max_val2

W1_quant = np.round(W1_float * scale_factor1).astype(int)
B1_quant = np.round(B1_float * scale_factor1).astype(int)
W2_quant = np.round(W2_float * scale_factor2).astype(int)
B2_quant = np.round(B2_float * scale_factor2).astype(int)

print("4. Saving files to 'mnist_mlp_data/'...")
os.makedirs("mnist_mlp_data", exist_ok=True)

# Save quantized integer files
np.savetxt("mnist_mlp_data/W1_matrix.txt", W1_quant, fmt='%d')
np.savetxt("mnist_mlp_data/B1_vector.txt", B1_quant, fmt='%d')
np.savetxt("mnist_mlp_data/W2_matrix.txt", W2_quant, fmt='%d')
np.savetxt("mnist_mlp_data/B2_vector.txt", B2_quant, fmt='%d')

# Save original float weights for full FP32 Neural Network baseline comparison
np.savetxt("mnist_mlp_data/W1_float.txt", W1_float, fmt='%.6f')
np.savetxt("mnist_mlp_data/B1_float.txt", B1_float, fmt='%.6f')
np.savetxt("mnist_mlp_data/W2_float.txt", W2_float, fmt='%.6f')
np.savetxt("mnist_mlp_data/B2_float.txt", B2_float, fmt='%.6f')

# Quantize sample image pixels to 8-bit integer range (0 to 255) for clean bit-slicing
for i in range(5):
    sample_img = X[i]
    sample_label = y[i]
    sample_quantized = np.round(sample_img * 255).astype(int) 
    
    # Save integer pixel input
    np.savetxt(f"mnist_mlp_data/sample_img_{i}_label_{sample_label}.txt", sample_quantized, fmt='%d')
    # Save normalized float pixel input (0.0 to 1.0)
    np.savetxt(f"mnist_mlp_data/sample_img_{i}_float.txt", sample_img, fmt='%.6f')
    
    print(f"   -> Saved Sample Image {i} (True Label: {sample_label})")

print("\nDone! All 32-bit safe MLP integer and float weight files generated successfully.")
