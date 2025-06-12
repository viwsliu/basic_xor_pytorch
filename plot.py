import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_decision_boundary(model):
  x1 = np.linspace(0, 1, 100)
  x2 = np.linspace(0, 1, 100)
  xx1, xx2 = np.meshgrid(x1, x2)
  grid = torch.tensor(np.c_[xx1.ravel(), xx2.ravel()], dtype=torch.float32)
  with torch.no_grad():
    preds = model(grid).reshape(xx1.shape)
    
  plt.contourf(xx1, xx2, preds, levels=50, cmap="coolwarm", alpha=0.6)
  plt.colorbar(label="Model output")

  X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
  y = torch.tensor([0, 1, 1, 0])
  for i, label in enumerate(y):
    color = "black" if label == 0 else "white"
    edge = "black"
    plt.scatter(X[i, 0], X[i, 1], c=color, edgecolors=edge, s=100, linewidths=1)

  plt.title("XOR Model Decision Boundary")
  plt.xlabel("x1")
  plt.ylabel("x2")
  plt.grid(True)
  os.makedirs("training_details", exist_ok=True)
  save_path = os.path.join("training_details", "xor_decision_boundary.png")
  plt.savefig(save_path)
  print(f"🖼️ Saved decision boundary plot to: {save_path}")
