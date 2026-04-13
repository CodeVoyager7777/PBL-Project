import torch
import torch.nn as nn
import torch.optim as optim
from model_3dcnn import CNN3D
from data_loader import load_dataset
from sklearn.model_selection import train_test_split

X, y = load_dataset("dataset")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print("Total samples:", len(y))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Stable:", (y == 0).sum().item())
print("Unstable:", (y == 1).sum().item())

model = CNN3D(num_classes=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
batch_size = 32

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (len(X_train) // batch_size)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)

    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("Training Complete")
