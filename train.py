import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop door elke zin in onze intentiepatronen
for intent in intents['intents']:
    tag = intent['tag']
    # voegt aan de taglijst toe
    tags.append(tag)
    for pattern in intent['patterns']:
        # symbolyseert elk woord in de zin
        w = tokenize(pattern)
        # voegt het woord aan de woordenlijst toe
        all_words.extend(w)
        # voegt aan het xy paar toe
        xy.append((w, tag))

# beperkt de invloed van woorden
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# verwijdert duplicaten en sorteert
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# maakt de trainingsdata
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: groep woorden voor elk pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss heeft alleen labels nodig, geen one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # ondersteuning van indexering zodat dataset [i] kan worden gebruikt om i-th sample te krijgen
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we kunnen len (dataset) gebruiken om de grootte te beperken
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# verliezen en optimaliseren
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# traint het model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # voorwaartse reactie
        outputs = model(words)
        # als y one-hot zou zijn, moeten we het toepassen
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # terug en optimaliseren
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
