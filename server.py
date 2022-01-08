pathToNet="./modelV3"

import torchvision.models as models
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True

import pyheif

#имена класса
classes = ['Hyundai SOLARIS',
 'KIA Rio',
 'Skoda Octavia',
 'Volkswagen Polo',
 'Volkswagen Tiguan']

#уверенности для отсечения неуверенных гипотез, собирались по датасету
classThresholds = np.array([0.60005697, 0.82278588, 0.77659323, 0.74224129, 0.67364123])

def load_image(pathToImage):
"""загрузка имени по пути

Keyword arguments:
pathToImage -- путь до файла

"""
  try:
    pilImage = Image.open(pathToImage)
    pilImage = pilImage.convert('RGB')
    return pilImage
  except:
    imRaw = pyheif.read(pathToImage)
    pilImage = Image.frombytes(mode=imRaw.mode, size=imRaw.size, data=imRaw.data)
    pilImage = pilImage.convert('RGB')
    return pilImage

#загрузка сети
net = nn.Sequential( *list(models.resnet50(pretrained=False).children())[:-1],\
                    nn.Flatten(1),
                    nn.Linear(in_features=2048, out_features=len(classes), bias=True)
                    )


net.load_state_dict(torch.load(pathToNet))
net.cuda()

net.eval()

def getImageClassProbabilities(image, shouldUseConfidences=False): 
  """получение вероятностей принадлежности к машине из списка

  Keyword arguments:
  image -- загруженная картинка
  shouldUseConfidences -- использовать уверенности для определения точности результата
  """

  image= image.convert('RGB')

  transform = transforms.Compose([transforms.Resize((1024,1024)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
                                ])
  
  out = net(transform(image).unsqueeze(0).cuda())
  probabilities = F.softmax(out, dim=-1)

  result = dict()
  result["probabilities"] = dict()
  for index in range(len(classes)):
    result["probabilities"][classes[index]] = round(probabilities[0][index].item(), 4)

  if shouldUseConfidences:
    confidencesPerClass = torch.sigmoid(out.detach().cpu()).numpy()
    mostProbClass = np.argmax(confidencesPerClass)

    result["confidence"] = bool(confidencesPerClass[0][mostProbClass] >= classThresholds[mostProbClass])

  return result


from flask import Flask, jsonify,request
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

@app.route("/")
def hello_world():
  return "hello_world"

@app.route("/recognition", methods=['POST'])
def recognition():
  json_data = request.get_json()
  im = Image.open(BytesIO(base64.b64decode(json_data['content'])))
  result = getImageClassProbabilities(im)
  return jsonify(result)

if __name__ == '__main__':
  app.run(host="0.0.0.0", port=5000)

