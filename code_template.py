'''
Project 3: Machine Learning C'25

This is a standard image classification task designed to help everyone get fimiliar with the entire preocess of building up a machine learning project and understanding the key concepts and functions of its different components.
We will use PyTorch for the whole project. The Program Template is given, both py version and ipynb version.
'''

################################################################################
# Import Packages
################################################################################
import os
import random

# Add matplotlib backend setting
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# data part
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import random_split, Subset

# model part
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# training & testing part
from tqdm.auto import tqdm

# evaluation & visualization part
# import sklearn
# import seaborn as sns
import sklearn.metrics as metrics
import seaborn as sns
import numpy as np

################################################################################
# Get Data
################################################################################
'''
FER2013 Dataset: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
-

For example code, we provide you how to load the training data into the dataloader which can be used to fit the model. 
Don't forget to download the data from Canvas and unzip it before you run the template.

Do check the file address in case you don't keep the file structure which works for the template.
-

After getting the dataset, can use dataset.classes to get the class names: ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'].
'''
# This is a command line code for Jupyter Notebook to unzip file
# Please change the first address to the zip file path
# The second path is for the images folder
# !unzip -q fer_2013_train.zip -d ./
# You can run "unzip -q fer_2013_train.zip -d ./" on your local machine to unzip the data


################################################################################
# Show Data
################################################################################
## If you want to show some image samples

## If you want to show some image samples

# Modify the image display part
def show_sample_images(image_dir, save_path='sample_images.png'):
    """
    Show and save sample images from each class
    """
    # Images from all classes
    class_folders = [folder for folder in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, folder))]

    # Random pick 1 image from every class
    num_images = len(class_folders)
    selected_images = []

    for class_name in class_folders:
        class_path = os.path.join(image_dir, class_name)
        image_files = [f for f in os.listdir(class_path) if f.endswith(".jpg")]
        if image_files:
            selected_image = random.choice(image_files)
            selected_images.append((os.path.join(class_path, selected_image), class_name))

    # Create figure
    fig, axes = plt.subplots(1, num_images, figsize=(20, 5))

    for i, (img_path, img_class) in enumerate(selected_images):
        image = Image.open(img_path)
        axes[i].imshow(image, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"{img_class}\n{os.path.basename(img_path)}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Sample images saved to {save_path}")

# Image dir
image_dir = "./fer_2013_train/fer_2013_train/train"

# Show and save sample images
try:
    show_sample_images(image_dir)
except Exception as e:
    print(f"Warning: Could not display sample images. Error: {e}")
    print("Continuing with training...")

################################################################################
# Transforms (*)
################################################################################
'''
For this part you need to try different transforms to get better classification results.
-

Torchvision provides lots of image preprocessing utilities, we can use them as ways to get data augmentation.
As you can see the samples above, the images have different sizes. We can use simple transforms to resize the PIL image and turn it into Tensor.
Also, data augmentation can be done with transforms. You can use it to produce a variety of images, including rotation, scaling, flipping, color adjustments, and more.
-

Please check PyTorch official website for transforms details: https://pytorch.org/vision/0.9/transforms.html
'''

# Usually we don't need augmentations in testing and also validation
# But we still need to resize the PIL image and transform it into Tensor
test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    # Reduce image size from 128x128 to 64x64
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # Add normalization to improve training stability
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# Use train_transforms to implement data augmentation
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    # Keep effective augmentations only
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

################################################################################
# Model (*)
################################################################################
'''
For this part you need to try different model structures to get better classification results. 
You can try different layer types, different number of layers, different activate functions, and etc.
-

The default classification accuracy is around 0.3 which is better than the random guess, but looking at the training process, the model basically didn't learn anything.
'''

class Classifier(nn.Module):
    def __init__(self, num_classes=7):
        super(Classifier, self).__init__()
        
        # Optimize model architecture for memory efficiency
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Calculate input features for fc layer
        self.fc_input_size = 64 * 8 * 8

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

################################################################################
# Configurations (*)
################################################################################
'''
Here is the part you can "control" the training process by setting your own hyperparameters, which might affect your model performance. 
Main hyperparameters include the data batch size, number of training epochs, the loss function, optimizer and etc.
-

Please try sets of experiments with different combinations of hyperparameters to determine which goes best with your model.
'''

# Check whether GPU is availbale and set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the model, and don't forget to put it on the device
model = Classifier().to(device)

###########################################################################################
###########################################################################################
# You can try differnet configurations below for training the model to get better results #
###########################################################################################
###########################################################################################


# #given by professor
# # The number of batch size
# batch_size = 64
#
# # The number of training epochs
# n_epochs = 5
#
# # Set up the criterion, we usually use cross-entropy as the measurement of classification performance
# criterion = nn.CrossEntropyLoss()
#
# # Initialize optimizer, you can try different hyperparameters or different types of optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

#configuration 1: Increase the number of epochs and use a different optimizer
# The number of batch size
# batch_size = 64
#
# # The number of training epochs
# n_epochs = 20  # Increased number of epochs for more training
#
# # Set up the criterion
# criterion = nn.CrossEntropyLoss()
#
# # Initialize optimizer with a lower learning rate and weight decay (for regularization)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Adding weight decay for better generalization

#configuration 2: Try using SGD optimizer with momentum
# The number of batch size
batch_size = 32  # Reduced batch size to save memory

# The number of training epochs
n_epochs = 15

# Set up the criterion
criterion = nn.CrossEntropyLoss()

# Initialize optimizer with Adam and weight decay for regularization
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Add learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='max', 
                            factor=0.5, patience=2, 
                            verbose=True)

#Configuration 3: Use a learning rate scheduler
# # The number of batch size
# batch_size = 32  # Smaller batch size for better generalization
#
# # The number of training epochs
# n_epochs = 15  # Moderate number of epochs
#
# # Set up the criterion
# criterion = nn.CrossEntropyLoss()
#
# # Initialize optimizer with a higher learning rate and add learning rate scheduler
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# # Learning rate scheduler that reduces learning rate when the validation loss plateaus
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

# Configuration 4: Use a different loss function and batch normalization
# # The number of batch size
# batch_size = 64
#
# # The number of training epochs
# n_epochs = 25  # Increased number of epochs to train longer
#
# # Set up the criterion with Label Smoothing Cross-Entropy
# criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
#
# # Initialize optimizer with Adam
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
#
# # Batch normalization in the model for better convergence
# model.add_module('batch_norm', nn.BatchNorm1d(num_features=model.num_features))  # Assuming model has `num_features`



################################################################################
# Dataloader 
################################################################################
'''
This is how PyTorch help you to load the data as a dataloader.
-

For the template, we use the ImageFolder provided by torchvision.datasets to read the data with the folder annotations. You can also write you own data class to read the images and labels.

For the template, we also use random_split from torch.utils.data to get the training and validation data. 
You can also try other data split methods such as train_test_split from sklearn and etc. The training and validation ratio can also be adjusted.

Do not use the train_transforms for the validation data, since you may have some data augmentation operations for training data.
- 

For more infomation about data and dataloader, please refer to the PyTorch website: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#datasets-dataloaders
'''

# Construct train datasets
# The argument "loader" tells how torchvision reads the data.
train_dir = "./fer_2013_train/fer_2013_train/train"

# We use ImageFolder to read the images and set the annotations for each image
# Do not use train_transforms for this step, since the validation data should not have data augmantation
full_train_dataset = ImageFolder(train_dir)

# Aligned with official dataset
original_class_to_idx = full_train_dataset.class_to_idx
desired_class_order = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

new_class_to_idx = {cls_name: i for i, cls_name in enumerate(desired_class_order)}
full_train_dataset.class_to_idx = new_class_to_idx

full_train_dataset.samples = [
    (img_path, new_class_to_idx[original_cls_name])
    for img_path, original_cls_idx in full_train_dataset.samples
    for original_cls_name, idx in original_class_to_idx.items() if idx == original_cls_idx
]

# Construct valid datasets
# You can also use other methods like train_test_split() to get the validation set

# 80% training，20% validation
train_ratio = 0.8
train_size = int(train_ratio * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

# random_split() by PyTorch
train_indices, val_indices = random_split(range(len(full_train_dataset)), [train_size, val_size])

train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_train_dataset, val_indices)

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# Set the transforms separately for training and validation data
train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = test_transforms

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# You are encouraged to show some train and validation data and label to know better about your data.

################################################################################
# Training Process 
################################################################################
'''
Show the whole training process including the validation part, and save model at the last step. 
Basically, you don't need to change anything for this part. However, you should know clearly about the whole process about how to train your model.
-

For more infomation about training a model, please refer to PyTorch website: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#training-with-pytorch
'''
# *Record the best validation acc to save the best model
best_acc = 0
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(n_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []
    
    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        logits = model(imgs)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()

        # Update the parameters with computed gradients.
        optimizer.step()
        
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        
        train_loss.append(loss.item())
        train_accs.append(acc.item())
        
        # Clear variables to free memory
        del imgs, labels, logits, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    # *
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(val_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        #break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    #*
    val_losses.append(valid_loss)
    val_accuracies.append(valid_acc)

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


    # save models
    if valid_acc > best_acc:
        save_path = "model_best.pth"
        print(f"Best model found at epoch {epoch+1}, saving model")
        torch.save(model.state_dict(), save_path) 
        best_acc = valid_acc

    # Update learning rate based on validation accuracy
    scheduler.step(valid_acc)


################################################################################
# Dataloader for Test
################################################################################
'''
We've used dataloader at the training phase, for testing part, the only difference is that we don't need to apply any data augmentation technique except for resize and ToTensor.
-

We keep part of the test data and we'll run your best performing model to determine its accuracy on our own test set.
'''

# This is a command line code for Jupyter Notebook to unzip file
# Please change the first address to the zip file path
# The second path is for the images folder
# !unzip -q fer_2013_test.zip -d ./
# You can run "unzip -q fer_2013_train.zip -d ./" on your local machine to unzip the data

# Construct test datasets
# The argument "loader" tells how torchvision reads the data.
test_dir = "./fer_2013_test/fer_2013_test/test"

# We use ImageFolder to read the images and set the annotations for each image
test_dataset = ImageFolder(test_dir, transform=test_transforms)

# Aligned with official dataset
original_class_to_idx = test_dataset.class_to_idx
desired_class_order = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

new_class_to_idx = {cls_name: i for i, cls_name in enumerate(desired_class_order)}
test_dataset.class_to_idx = new_class_to_idx

test_dataset.samples = [
    (img_path, new_class_to_idx[original_cls_name])
    for img_path, original_cls_idx in test_dataset.samples
    for original_cls_name, idx in original_class_to_idx.items() if idx == original_cls_idx
]

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

################################################################################
# Testing Process 
################################################################################
'''
This is the testing phase for verifying your model's performance. 
We need to load the best model you saved during the training phase with trained parameters. 
After we get the predictions by the model and true labels, we can use them to do the evaluations.
-

For more infomation about testing a model, please refer to PyTorch website: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#training-with-pytorch
'''

save_path = "model_best.pth"
model_best = Classifier().to(device)
model_best.load_state_dict(torch.load(save_path))

model_best.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for data, labels in tqdm(test_loader):
        model_output = model_best(data.to(device))
        predicted_label = torch.argmax(model_output, dim=1).cpu().numpy()

        predictions.extend(predicted_label)
        true_labels.extend(labels)

################################################################################
# Evaluation & Visualizations (*)
################################################################################

def save_training_plot(train_accuracies, val_accuracies, save_path='training_plot.png'):
    """Save training and validation accuracy plot"""
    epochs_range = range(len(train_accuracies))
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training plot saved to {save_path}")

def save_confusion_matrix(true_labels, predictions, class_names, save_path='confusion_matrix.png'):
    """Save confusion matrix visualization"""
    confusion_matrix = metrics.confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
    
    # Print classification report
    print("\nClassification Report:")
    print(metrics.classification_report(true_labels, predictions, target_names=class_names))

def save_misclassified_examples(test_dataset, predictions, true_labels, class_names, save_path='misclassified_examples.png'):
    """Save visualization of misclassified examples"""
    misclassified_indices = np.where(np.array(predictions) != np.array(true_labels))[0]
    
    if len(misclassified_indices) == 0:
        print("No misclassified examples found!")
        return
        
    n_examples = min(3, len(misclassified_indices))
    misclassified_images = []

    for idx in misclassified_indices[:n_examples]:
        img_path, true_label = test_dataset.samples[idx]
        predicted_label = predictions[idx]
        image = Image.open(img_path).convert('L')
        misclassified_images.append((image, true_label, predicted_label))

    plt.figure(figsize=(15, 5))
    for i, (img, true_label, predicted_label) in enumerate(misclassified_images):
        plt.subplot(1, 3, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {class_names[true_label]}\nPred: {class_names[predicted_label]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Misclassified examples saved to {save_path}")

# Save training plot
save_training_plot(train_accuracies, val_accuracies)

# Save confusion matrix
save_confusion_matrix(true_labels, predictions, desired_class_order)

# Save misclassified examples
save_misclassified_examples(test_dataset, predictions, true_labels, desired_class_order)