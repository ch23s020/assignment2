
#%pip install wandb

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import wandb

def train(config):
    # Data augmentation and normalization
    if config.data_augmentation == 'yes':
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    else:
        train_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Loading dataset from google drive
    train_data = datasets.ImageFolder(root='/content/drive/MyDrive/inaturalist_12K/train', transform=train_transforms)

    test_data = datasets.ImageFolder(root='/content/drive/MyDrive/inaturalist_12K/test', transform=test_transforms)

    # Spliting train_data into train and validation as mentioned 20% in question
    train_size = int(0.8 * len(train_data))

    val_size = len(train_data) - train_size

    train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

    # DataLoader with multiprocessing (Here Setting num_workers =2 to simultaneously do the processing and loading to decrase the wait time)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)

    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    test_loader = DataLoader(dataset=test_data, batch_size=config.batch_size, shuffle=False, num_workers=2)
#(Tried different values of num_workers =4 but wandb crashesh for values greater than 2 )

    class CNN(nn.Module):

        def __init__(self, num_classes, num_filters, filter_size, activation, filter_org, use_batchnorm, use_dropout, num_neurons):

            super(CNN, self).__init__()

            self.num_classes = num_classes

            self.num_filters = num_filters

            self.filter_size = filter_size

            self.activation = activation

            self.filter_org = filter_org

            self.use_batchnorm = use_batchnorm

            self.use_dropout = use_dropout

            self.num_neurons = num_neurons

            # Defining the Convolution layers (CNN )
            layers = []

            in_channels = 3 # Given image is of RGB type

            prev_output_size = 224  # Initial image size, adjust as necessary will be helpful in fine tuning over other dataset

            for i in range(len(num_filters)):

                layers.append(nn.Conv2d(in_channels, num_filters[i], filter_size[i]))

                if use_batchnorm:
                    layers.append(nn.BatchNorm2d(num_filters[i]))

                if activation == 'ReLU':
                    layers.append(nn.ReLU())

                elif activation == 'GELU':
                    layers.append(nn.GELU())

                elif activation == 'SiLU':
                    layers.append(nn.SiLU())

                elif activation == 'Mish':
                    layers.append(nn.Mish())

                layers.append(nn.MaxPool2d(2, 2)) # Adding maxpooling functionality


                #Adding droput

                if use_dropout:
                    layers.append(nn.Dropout(0.2))

                in_channels = num_filters[i]

                # Calculate the output size of this layer

                prev_output_size = (prev_output_size - filter_size[i] + 1) // 2

            self.conv_layers = nn.Sequential(*layers)

            #Dense Layer Calculation:

            # Calculate the input size to the fully connected layers 
            self.fc_input_size = in_channels * prev_output_size * prev_output_size

            # Dense layers
            self.fc = nn.Linear(self.fc_input_size, num_neurons)

            self.fc2 = nn.Linear(num_neurons, num_classes)

        def forward(self, x):

            x = self.conv_layers(x)

            x = x.view(x.size(0), -1)

            x = self.fc(x)

            x = self.fc2(x)

            return x

    # Now Initializing the model

    model = CNN(num_classes=10, num_filters=config.num_filters, filter_size=config.filter_size,
                
                activation=config.activation, filter_org=config.filter_org,

                use_batchnorm=config.use_batchnorm, use_dropout=config.use_dropout,

                num_neurons=config.num_neurons)

    # Loss function and optimizer

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Training loop

    for epoch in range(config.num_epochs):  # To use num_epochs from config

        model.train()

        running_loss = 0.0

        correct_train = 0

        total_train = 0

        for batch_idx, (data, targets) in enumerate(train_loader):


            # Forward_Prop:

            outputs = model(data)

            loss = criterion(outputs, targets)

            # Backward_Prop:

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            #appending loss
            running_loss += loss.item()

            # Calculating training accuracy:

            _, predicted = torch.max(outputs.data, 1)

            total_train += targets.size(0)

            correct_train += (predicted == targets).sum().item()

            # Print training accuracy after each batch for better visibility:

            train_accuracy = 100 * correct_train / total_train

            print(f'Epoch [{epoch+1}/{config.num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Train Acc: {train_accuracy:.2f}%')

        # Validation loop (Printing after each epoch)

        model.eval()

        with torch.no_grad():

            correct_val = 0

            total_val = 0

            val_loss = 0.0

            for data, targets in val_loader:

                outputs = model(data)

                val_loss += criterion(outputs, targets).item()

                # Calculate validation accuracy

                _, predicted = torch.max(outputs.data, 1)

                total_val += targets.size(0)

                correct_val += (predicted == targets).sum().item()

            val_accuracy = 100 * correct_val / total_val

            print(f'Epoch [{epoch+1}/{config.num_epochs}], Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.2f}%')

        # Log validation accuracy to Wandb
        wandb.log({'val_accuracy': val_accuracy})


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_filters', type=int, nargs='+', help='Number of filters for each convolutional layer')

    parser.add_argument('--filter_size', type=int, nargs='+', help='Size of filters for each convolutional layer')

    parser.add_argument('--activation', type=str, help='Activation function for convolutional layers')

    parser.add_argument('--use_batchnorm', type=int, help='Whether to use batch normalization (0 or 1)')

    parser.add_argument('--use_dropout', type=int, help='Whether to use dropout (0 or 1)')

    parser.add_argument('--lr', type=float, help='Learning rate for optimizer')

    parser.add_argument('--num_epochs', type=int, help='Number of epochs for training')

    parser.add_argument('--filter_org', type=str, default='same', help='Filter organization')

    parser.add_argument('--data_augmentation', type=str, help='Whether to use data augmentation (yes or no)')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')

    parser.add_argument('--num_neurons', type=int, help='Number of neurons in fully connected layers')

    parser.add_argument('--learning_algorithm', type=str, help='Learning algorithm (e.g., Adam, SGD)')

    args = parser.parse_args()

    wandb.init(config=args)

    train(args)

    wandb.finish()
