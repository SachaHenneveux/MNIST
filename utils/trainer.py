import time
import torch
from models.mnist_model import MODEL_PATH

def train_model(self, epochs, train_loader):
        self.train()

        for epoch in range(epochs):
            start_time = time.time()
            running_loss = 0.0
            total_batches = 0

            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                total_batches += 1

                if (i + 1) % 8 == 0 or (i + 1) == len(train_loader):
                    print(f"\rEpochs {epoch + 1}/{epochs} | Lot {i + 1}/{len(train_loader)} | Loss : {loss.item():.4f}", end='')

            
            avg_loss = running_loss / len(train_loader)
            epoch_time = time.time() - start_time

            print("\n")
            print("-" * 60)
            print(f"Epochs {epoch + 1}/{epochs} finish | Average Loss : {avg_loss:.4f} | Time : {epoch_time:.2f} seconds")
            print("-" * 60)
            self.scheduler.step()

        model_path = MODEL_PATH
        print('Training finished, saving model to :', model_path)
        torch.save(self.state_dict(), model_path)


def eval_model(self, test_loader):
    self.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on {total} images is : {100 * correct / total:.2f}%')
