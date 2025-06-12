from data import get_xor_data
from model import XORNet
from train import train
from plot import plot_decision_boundary
import torch
import os

def main():
  X, y = get_xor_data()
  model = XORNet()
  model_file = "xor_model.pth"

  if os.path.exists(model_file):
    print("Loading saved model")
    model.load_state_dict(torch.load(model_file))
    model.eval()
  else:
    print("Training new model")
    train(model, X, y)
    torch.save(model.state_dict(), model_file)
  print("\nPredictions:")
  with torch.no_grad():
    preds = model(X)
    print(preds.round())
  plot_decision_boundary(model)

if __name__ == "__main__":
  main()
