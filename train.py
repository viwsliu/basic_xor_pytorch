import torch
import torch.nn as nn
import torch.optim as optim

def train(model, X, y, epochs=5000, lr=0.1):
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
  for epoch in range(epochs):
    # forward pass
    pred = model(X)
    loss = criterion(pred, y)
    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
      print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
