# DogIA

Criando uma aplicação simples de classificação de imagens usando uma rede neural pré-treinada, utiliza o PyTorch para carregar um modelo de rede neural pré-treinado (ResNet50), baixar um arquivo com os rótulos das classes do ImageNet. O objetivo é identificar qualquer raça de cachorro através de uma imagem.

<h2> Bibliotecas </h2>

```python 

import json # manipular arquivos JSON
import torch # lidar com operações em tensores
import torchvision.models as models #  carregar modelos pré-treinados e transformar imagens
import torchvision.transforms as transforms # contém várias transformações que podem ser aplicadas em imagens usando o PyTorch
import requests # baixar arquivos JSON da web
