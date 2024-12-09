import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Carga del conjunto de datos MNIST
train = datasets.MNIST('', train=True, download=True, 
                       transform=transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST('', train=False, download=True, 
                      transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

# Definición de la clase de la red neuronal
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)  # Capa totalmente conectada 1
        self.fc2 = nn.Linear(64, 64)       # Capa totalmente conectada 2
        self.fc3 = nn.Linear(64, 64)       # Capa totalmente conectada 3
        self.fc4 = nn.Linear(64, 10)       # Capa de salida con 10 clases

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activación ReLU después de la capa 1
        x = F.relu(self.fc2(x))  # Activación ReLU después de la capa 2
        x = F.relu(self.fc3(x))  # Activación ReLU después de la capa 3
        x = self.fc4(x)          # Salida sin activación en la capa final
        return F.log_softmax(x, dim=1)  # Activación Softmax logarítmica para clasificación

# Inicialización de la red neuronal
net = Net()

# Definición de la función de pérdida y el optimizador
loss_function = nn.CrossEntropyLoss()  # Función de pérdida de entropía cruzada
optimizer = optim.Adam(net.parameters(), lr=0.001)  # Optimizador Adam con tasa de aprendizaje 0.001

# Entrenamiento de la red neuronal
for epoch in range(3):  # 3 épocas
    for data in trainset:  # Iterar sobre los lotes de datos
        X, y = data  # Características (X) y etiquetas (y)
        net.zero_grad()  # Reinicia los gradientes
        output = net(X.view(-1, 28 * 28))  # Aplana las imágenes y realiza la pasada hacia adelante
        loss = F.nll_loss(output, y)  # Calcula la pérdida
        loss.backward()  # Propagación hacia atrás
        optimizer.step()  # Ajusta los pesos
    print(f"Epoch {epoch+1} Loss: {loss.item()}")

# Prueba de la red neuronal
correct = 0
total = 0

with torch.no_grad():  # Desactiva el cálculo de gradientes
    for data in testset:
        X, y = data
        output = net(X.view(-1, 28 * 28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:  # Comprueba si la predicción es correcta
                correct += 1
            total += 1

print("Accuracy:", round(correct / total, 3))

# Mostrar una imagen de ejemplo y su predicción
sample = next(iter(testset))
image, label = sample[0][0], sample[1][0]
plt.imshow(image.view(28, 28), cmap="gray")
plt.show()
print("Predicted Label:", torch.argmax(net(image.view(-1, 28 * 28))[0]))
