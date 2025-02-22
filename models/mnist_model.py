import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os

LEARNING_RATE = 0.001
MODEL_PATH = "mnist_model.pth"

class MNISTModel(nn.Module):
    def __init__(self, resume_training=False):
        super(MNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.fc1 = nn.Linear(128 * 7 * 7 , 16 * 7 * 7)
        self.fc2 = nn.Linear(16 * 7 * 7, 7 * 7)
        self.fc3 = nn.Linear(7 * 7, 10)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.drop1 = nn.Dropout(p=0.25)
        self.batchNorme = nn.BatchNorm2d(64)
        self.batchNorme2 = nn.BatchNorm2d(128)
    
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=LEARNING_RATE)
        self.relu = nn.ReLU()
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.7)
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"Device : {self.device}")
        self.to(self.device)

        if resume_training and os.path.exists(MODEL_PATH):
            self.load_weights(MODEL_PATH)
            print(f"Resuming training from weights at {MODEL_PATH}")
        else:
            print("Starting training from scratch.")

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(self.relu(self.conv2(x)))
        x = self.batchNorme(x)
        x = self.relu(self.conv3(x))
        x = self.pool2(x)
        x = self.batchNorme2(x)

        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.drop1(x)
        x = self.fc3(x)
        return x
        
    def load_weights(self, model_path):
        self.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        self.eval()