from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTConfig
from PIL import Image
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt

# Load the local config file
config = ViTConfig.from_json_file('config.json')

# Load the pre-trained ViT model and feature extractor using the local config
feature_extractor = ViTFeatureExtractor.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
model = ViTForImageClassification(config=config)
pretrained_dict=torch.load('cifar10_checkpoint.bin', map_location='cpu')
print("Pretrained Dict",pretrained_dict.keys())
# model.load_state_dict(torch.load(pretrained_dict, map_location=torch.device('cpu')))

# Download an image for testing
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog10.png'
image = Image.open(requests.get(url, stream=True).raw)
inputs = feature_extractor(images=image, return_tensors="pt")

# Make a prediction
outputs = model(**inputs)
preds = outputs.logits.argmax(dim=1)

# Print the predicted class
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]
print("Predicted class:", classes[preds[0]])

# Load the CIFAR-10 testing dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
])

test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

# Evaluate the model on the testing dataset
model.eval()
correct = 0
total = 0
predictions = []

with torch.no_grad():
    for i in range(len(test_dataset)):
        inputs, labels = test_dataset[i]
        inputs = feature_extractor(images=inputs, return_tensors="pt")
        outputs = model(**inputs)
        _, predicted = torch.max(outputs.logits, 1)
        predictions.append(predicted.item())
        total += 1
        correct += (predicted == labels).item()

accuracy = correct / total
print(f'Accuracy on CIFAR-10: {accuracy * 100:.2f}%')

