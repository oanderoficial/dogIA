# DogIA

Criando uma aplicação simples de classificação de imagens usando uma rede neural pré-treinada, utiliza o PyTorch para carregar um modelo de rede neural pré-treinado (ResNet50), baixar um arquivo com os rótulos das classes do ImageNet. O objetivo é identificar qualquer raça de cachorro através de uma imagem.

```
python -m pip install --upgrade pip 
```

```
python -m pip install --upgrade setuptools
```

```
pip install torch 
```

```
pip install torchvision
```
<h2> Bibliotecas </h2>

```python 

import json # manipular arquivos JSON
import torch # lidar com operações em tensores
import torchvision.models as models #  carregar modelos pré-treinados e transformar imagens
import torchvision.transforms as transforms # contém várias transformações que podem ser aplicadas em imagens usando o PyTorch
import requests # baixar arquivos JSON da web
```

<strong> Carregar o modelo ResNet50 pré-treinado </strong>

```python
modelo = models.resnet50(pretrained=True)
```

<strong> Carregar o arquivo com os rótulos das classes </strong> 
```python
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
response = requests.get(url)
rotulos = json.loads(response.text)
```

<strong> Carregar a imagem e transformá-la </strong>
```python
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
```
