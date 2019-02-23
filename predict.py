# Imports here
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import os
import json
from torchvision import datasets, transforms, models
from PIL import Image
from torch.autograd import Variable
# Imports here

means = [0.485, 0.456, 0.406]
deviations = [0.229, 0.224, 0.225]


def parse_args():
  parser = argparse.ArgumentParser(
    description="Trains a network on a dataset of images and saves the model to a checkpoint")
  parser.add_argument('data_dir', action="store", help='the data directory to work')
  parser.add_argument('check_point', action="store", type=str, help='the save file to load')
  parser.add_argument('--topk', default=3, type=int, help='the num to show of possible candidates from top')
  parser.add_argument('--json_dir', default='./cat_to_name.json', type=str, help='select the json file path')
  parser.add_argument('--gpu', action='store_true', help='set the gpu mode')
  args = parser.parse_args()

  return args

def process_image(image):
  """ Scales, crops, and normalizes a PIL image for a PyTorch model,
    #   # TODO: Process a PIL image for use in a PyTorch model

    #   # TODO : First, resize the images where the shortest side is 256 pixels,
    #   #        keeping the aspect ratio. This can be done with the thumbnail or resize methods

    #   resize_to = 256
    #   crop_to = 224

    #   # resize
    #   image_width, image_height = image.size
    #   shorter = image_width if image_width < image_height else image_height
    #   ratio = resize_to / shorter
    #   resized_width = int(image_width * ratio)
    #   resized_height = int(image_height * ratio)
    #   image = image.resize((resized_width, resized_height))

    #   image = image.crop((c / 2 for c in ((resized_width - crop_to), (resized_height - crop_to),
    #                                       (resized_width + crop_to), (resized_height + crop_to))))

    #   # TODO : Color channels of images are typically encoded as integers 0-255,
    #   #        but the model expected floats 0-1.
    #   #        You'll need to convert the values.
    #   #        It's easiest with a Numpy array, which you can get from a PIL image like so np_image = np.array(pil_image).

    #   # 0-255 to 0-1
    #   image = np.array(image)
    #   image = image / 255.

    #   # Nomalization
    #   mean = np.array(means)
    #   std = np.array(deviations)
    #   image = (image - mean) / std

    #   # Transpose
    #   image = np.transpose(image, (2, 0, 1))

    #   return image.astype(np.float32)
  """
  preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  return preprocess(image)


def imshow(image, ax=None, title=None):
  """Imshow for Tensor."""
  if ax is None:
    fig, ax = plt.subplots()

  # PyTorch tensors assume the color channel is the first dimension
  # but matplotlib assumes is the third dimension
  image = image.transpose((1, 2, 0))

  # Undo preprocessing
  mean = np.array(means)
  std = np.array(deviations)
  image = std * image + mean

  # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
  image = np.clip(image, 0, 1)

  ax.imshow(image)

  return ax


def predict(image_path, device, model, label_map, topk=5):
  ''' Implementation of the code to predict the class from an image file'''

  image = Image.open(image_path)
  image = process_image(image)
  image = np.expand_dims(image, 0)
  image = torch.from_numpy(image)

  model.eval()

  image = Variable(image).to(device)

  logits = model.forward(image)
  result = F.softmax(logits, dim=1)
  top_probs, top_labels = result.cpu().topk(topk)
    
  top_probs = top_probs.detach().numpy().tolist()[0]
  top_labels = top_labels.detach().numpy().tolist()[0]
    
  idx_to_class = {val: key for key, val in model.class_to_idx.items()}
  top_labels = [idx_to_class[lab] for lab in top_labels]
  top_flowers = [label_map[lab] for lab in top_labels]

  return top_flowers, top_probs



# TODO: Display an image along with the top 5 classes
def view_classify(image, ps, classes):
  ''' Function for viewing an image and it's predicted classes.
  '''
  num_classes = len(ps)
  ps = np.array(ps)
  image = image.transpose((1, 2, 0))

  # Undo preprocessing
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  image = std * image + mean

  # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
  image = np.clip(image, 0, 1)

  fig, (ax1, ax2) = plt.subplots(figsize=(10, 6), ncols=2)
  ax1.imshow(image)
  ax1.axis('off')
  ax2.barh(np.arange(num_classes), ps)
  # ax2.set_aspect(0.1)
  ax2.set_yticks(np.arange(num_classes))
  ax2.set_yticklabels(np.arange(num_classes).astype(int), size='large');
  ax2.set_title('Class Probability')
  ax2.set_xlim(0, 1.1)
  ax2.set_yticklabels(classes)
  fig.subplots_adjust(wspace=.6)


def is_exist_savefile(save_dir):
  return os.path.isfile(save_dir)


def main():
  args = parse_args()

  data_dir = args.data_dir
  check_point = args.check_point
  topk = args.topk
  gpu = args.gpu
  json_dir = args.json_dir
    
  print('=' * 10 + 'Params' + '=' * 10)
  print('Data dir:      {}'.format(data_dir))
  print('Check point:   {}'.format(check_point))
  print('Top k:         {}'.format(topk))
  print('Gpu:           {}'.format(gpu))
  print('Json:          {}'.format(json_dir))
    
  if not is_exist_savefile(data_dir):
    raise Exception('Not found any file.. please check the file path.')

  if not check_point:
    raise Exception('Check point must be required. please specify the correct check point path..')

  device = torch.device("cuda" if gpu else "cpu")
  with open(json_dir, 'r') as f:
    label_map = json.load(f)
    
  image = Image.open(data_dir)
  image = process_image(image)

  print('Now Loading the check point..')
  check_point = torch.load(check_point, map_location=lambda storage, loc: storage)
  arch = check_point['arch']
  model = None
  if arch == 'VGG':
    model = models.vgg16(pretrained=True)
  else:
    model = models.densenet121(pretrained=True)
    
  model.classifier = check_point['classifier']
  model.load_state_dict(check_point['model'])
  model.class_to_idx = check_point['class_to_idx']

  model.to(device)
  model.eval()

  top_flowers, top_probs = predict(data_dir, device, model, label_map, topk=5)

  final_result = []
  for i in range(topk):
    final_result.append('%s : %d%%' % (top_flowers[i], int(float(top_probs[i])*100)))
    
  print(final_result)
    
if __name__ == '__main__':
  main()
# python predict.py ./rose.jpg vgg_5120_1024.pth
# python predict.py ./rose.jpg densenet_512_512.pth