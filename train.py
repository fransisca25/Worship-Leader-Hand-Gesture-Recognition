import os

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import matplotlib.pyplot as plt

from tool.dir_generator import codes, no_sequences, sequence_len
from dataset_loader import create_dataloaders
from transformer import HandGestureTransformer


# Later use cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = create_dataloaders(
    data_path="MP_Data",
    codes=codes,
    no_sequences=no_sequences,
    sequence_len=sequence_len,
    batch_size=8,
    train_split=0.8
)

# TEST LOADER (DEBUG PRINT)
for X_batch, y_batch in train_loader:
    print("X_batch shape:", X_batch.shape)
    print("y_batch shape:", y_batch.shape)
    print("X dtype:", X_batch.dtype)
    print("y dtype:", y_batch.dtype)
    break

model = HandGestureTransformer(
    input_dim=63,
    d_model=64,
    d_ff=128,
    num_heads=2,
    dropout=0.1,
    num_layers=2,
    num_classes=len(codes),
    max_seq_len=sequence_len
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# save trained model
SAVE_DIR = "trained_model"
os.makedirs(SAVE_DIR, exist_ok=True)

log_path = os.path.join(SAVE_DIR, "training_log.txt")
model_path = os.path.join(SAVE_DIR, "wl_hand_gesture_transformer.pth")
plot_path = os.path.join(SAVE_DIR, "training_plot.png")

# -------------------------- TRAIN AND TEST -----------------------------
train_losses = []
train_accuracies = []

test_losses = []
test_accuracies = []

# define number of epochs
epochs = 50

with open(log_path, "w") as log_file:
    log_file.write("epoch, train_loss, train_accuracy, test_loss, test_accuracy\n")

    for epoch in range(epochs):
        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]"):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

        avg_loss = total_loss / len(train_loader)
        train_accuracy = correct / total

        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)

        # test
        model.eval()

        test_total_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for X_batch, y_batch in tqdm(test_loader):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                test_total_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                test_correct += (predictions == y_batch).sum().item()
                test_total += y_batch.size(0)

        avg_test_loss = test_total_loss / len(test_loader)
        test_accuracy = test_correct / test_total

        test_losses.append(avg_test_loss)
        test_accuracies.append(test_accuracy)

        log_text = (
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {avg_loss:.4f} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Test Loss: {avg_test_loss:.4f} "
            f"Test Acc: {test_accuracy:.4f}\n"
        )

        print(log_text)

        log_file.write(
            f"{epoch + 1}, {avg_loss:.4f}, {train_accuracy:.4f}, "
            f"{avg_test_loss:.4f}, {test_accuracy:.4f}\n"
        )

        log_file.flush()


# save model
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "codes": codes,
    "sequence_len": sequence_len,
    "input_dim": 63,
    "d_model": 64,
    "d_ff": 128,
    "num_heads": 2,
    "dropout": 0.1,
    "num_layers": 2,
    "num_classes": len(codes)
}, model_path)

print(f"Model saved to {model_path}!\n")

# generate plot
plt.figure(figsize=(10, 5))

plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, epochs + 1), test_losses, label="Test Loss")
plt.plot(range(1, epochs + 1), train_accuracies, label="Train Accuracy")
plt.plot(range(1, epochs + 1), test_accuracies, label="Test Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training and Evaluation Loss/Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(plot_path)
plt.close()

print(f"Training plot saved to {plot_path}")
print(f"Training log saved to {log_path}")



