import json # manipular arquivos JSON
import torch # lidar com operações em tensores
import torchvision.models as models #  carregar modelos pré-treinados e transformar imagens
import torchvision.transforms as transforms # contém várias transformações que podem ser aplicadas em imagens usando o PyTorch
import requests # baixar arquivos JSON da web
from PIL import Image # manipular imagens.

# Carregar o modelo ResNet50 pré-treinado
modelo = models.resnet50(pretrained=True)

# Carregar o arquivo com os rótulos das classes
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
response = requests.get(url)
rotulos = json.loads(response.text)

# Carregar a imagem e transformá-la
caminho = input("Digite o nome da imagem:")
imagem = Image.open(caminho)
transform = transforms.Compose([
    transforms.Resize(256), #  redimensiona a imagem para que o menor lado tenha tamanho 256 pixels.
    transforms.CenterCrop(224), # corta a imagem centralmente para que tenha tamanho 224x224 pixels.
    transforms.ToTensor(), # : converte a imagem em um tensor do PyTorch.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]) # normaliza os valores dos pixels da imagem usando a média e o desvio padrão da ImageNet, que são valores predefinidos.
])
img_tensor = transform(imagem)
img_tensor = img_tensor.unsqueeze(0) #usado para adicionar uma dimensão extra ao tensor

# Classificar a imagem
modelo.eval()
with torch.no_grad():
    outputs = modelo(img_tensor)
_, predicted = torch.max(outputs, 1)

# Obter a classe prevista e o rótulo correspondente
classe = predicted.item()
rotulo = rotulos[str(classe)][1]

# Imprimir a classe prevista e o rótulo correspondente
print(f"Classe prevista: {classe}")
print(f"Rótulo: {rotulo}")