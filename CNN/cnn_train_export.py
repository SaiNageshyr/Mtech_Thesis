import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# -----------------------------------------------------------------------------
# 1. Safe Quantization Aware Training (QAT) Mechanisms
# -----------------------------------------------------------------------------
# Reduced from 2,147,483,647 to 10000.0 to prevent 64-bit float overflow 
# during NgSPICE analog accumulation and Verilog ALU processing.
QAT_SCALE = 10000.0 

class FakeQuantizeSafe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x * QAT_SCALE) / QAT_SCALE

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def qat_safe(x):
    return FakeQuantizeSafe.apply(x)

# -----------------------------------------------------------------------------
# 2. Define the Modern Custom 3-Layer CNN Architecture
# -----------------------------------------------------------------------------
class ModernCustomCNN_QAT(nn.Module):
    def __init__(self):
        super(ModernCustomCNN_QAT, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=0)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=0)
        self.relu2 = nn.ReLU()
        
        self.fc = nn.Linear(8 * 6 * 6, 10)

    def forward(self, x):
        x = qat_safe(x)
        w1_qat = qat_safe(self.conv1.weight)
        b1_qat = qat_safe(self.conv1.bias)
        x = self.relu1(nn.functional.conv2d(x, w1_qat, b1_qat, stride=2))
        
        x = qat_safe(x)
        w2_qat = qat_safe(self.conv2.weight)
        b2_qat = qat_safe(self.conv2.bias)
        x = self.relu2(nn.functional.conv2d(x, w2_qat, b2_qat, stride=2))
        
        x = x.view(x.size(0), -1) 
        x = qat_safe(x)
        
        w3_qat = qat_safe(self.fc.weight)
        b3_qat = qat_safe(self.fc.bias)
        x = nn.functional.linear(x, w3_qat, b3_qat)
        return x

def export_matrix(filename, matrix, is_integer=False):
    fmt = '%d' if is_integer else '%.6f'
    np.savetxt(filename, matrix, fmt=fmt)

# -----------------------------------------------------------------------------
# 3. Train the Model & Export Weights
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("mnist_cnn_data", exist_ok=True)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
    
    model = ModernCustomCNN_QAT()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    print("Training Safe QAT CNN for 1 Epoch...")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/len(train_loader) | Loss: {loss.item():.4f}")
            if batch_idx == 400: break 
            
    # -----------------------------------------------------------------------------
    # 4. Export Im2Col Unrolled Weights for Hardware Crossbar
    # -----------------------------------------------------------------------------
    model.eval()
    
    # --- LAYER 1 CONV EXPORT ---
    w1 = model.conv1.weight.detach().numpy()
    b1 = model.conv1.bias.detach().numpy()
    w1_flat = w1.reshape(w1.shape[0], -1).T 
    
    export_matrix("mnist_cnn_data/W1_float.txt", w1_flat)
    export_matrix("mnist_cnn_data/B1_float.txt", b1)
    export_matrix("mnist_cnn_data/W1_matrix.txt", np.round(w1_flat * QAT_SCALE).astype(np.int64), True)
    export_matrix("mnist_cnn_data/B1_vector.txt", np.round(b1 * QAT_SCALE).astype(np.int64), True)

    # --- LAYER 2 CONV EXPORT ---
    w2 = model.conv2.weight.detach().numpy()
    b2 = model.conv2.bias.detach().numpy()
    w2_flat = w2.reshape(w2.shape[0], -1).T
    
    export_matrix("mnist_cnn_data/W2_float.txt", w2_flat)
    export_matrix("mnist_cnn_data/B2_float.txt", b2)
    export_matrix("mnist_cnn_data/W2_matrix.txt", np.round(w2_flat * QAT_SCALE).astype(np.int64), True)
    export_matrix("mnist_cnn_data/B2_vector.txt", np.round(b2 * QAT_SCALE).astype(np.int64), True)

    # --- LAYER 3 FC EXPORT ---
    w3 = model.fc.weight.detach().numpy()
    b3 = model.fc.bias.detach().numpy()
    w3_flat = w3.T 
    
    export_matrix("mnist_cnn_data/W3_float.txt", w3_flat)
    export_matrix("mnist_cnn_data/B3_float.txt", b3)
    export_matrix("mnist_cnn_data/W3_matrix.txt", np.round(w3_flat * QAT_SCALE).astype(np.int64), True)
    export_matrix("mnist_cnn_data/B3_vector.txt", np.round(b3 * QAT_SCALE).astype(np.int64), True)

    # --- EXPORT 5 SAMPLE IMAGES ---
    sample_images, sample_labels = next(iter(train_loader))
    
    for i in range(5):
        img_2d = sample_images[i].numpy().squeeze() 
        label = sample_labels[i].item()
        
        np.savetxt(f"mnist_cnn_data/sample_img_{i}_float.txt", img_2d, fmt='%.6f')
        np.savetxt(f"mnist_cnn_data/sample_img_{i}_label_{label}.txt", np.round(img_2d * QAT_SCALE).astype(np.int64), fmt='%d')
        
    print("Export Complete: Safe QAT CNN weights unrolled and 5 sample images saved to ./mnist_cnn_data/")
