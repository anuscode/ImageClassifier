# Imports here

import argparse
import os
import os.path
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models


def parse_args():
  parser = argparse.ArgumentParser(
    description="Trains a network on a dataset of images and saves the model to a checkpoint")
  parser.add_argument('data_dir', action="store")
  parser.add_argument('--save_dir', default=None, type=str, help='set the save name')
  parser.add_argument('--arch', default='vgg', type=str, help='choose the model architecture')
  parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate')
  parser.add_argument('--hidden_units', default=None, nargs='+', type=int, help='list of integers, the sizes of the hidden layers')
  parser.add_argument('--epochs', default=2, type=int, help='num of training epochs')
  parser.add_argument('--gpu', action='store_true', help='set the gpu mode')
  parser.add_argument('--dropout', default=0.5, type=float, help='set dropout rate')
  parser.add_argument('--full_model_update', default=False, type=bool, help='full model update')
  args = parser.parse_args()

  return args

def get_architecture(arch):
  #Building model

  if arch in ['vgg16', 'vgg']:
    model = models.vgg16(pretrained=True)
    input_size = 25088
  elif arch in ['densenet121', 'densenet']:
    model = models.densenet121(pretrained=True)
    input_size = 1024
  else:
    raise ValueError('Only vgg16 and densenet121 are supported..')

  # Freeze the parameters for back propagation
  for param in model.parameters():
    param.requires_grad = False

  return model, input_size


def get_classifier(input_size, hidden_units, output_units, drop_p=None):
  hidden_units.append(output_units)

  if input_size < hidden_units[0]:
    raise ValueError('Please choose a hidden_unit lower than ' + str(input_size))

  # Add the first layer, input to a hidden layer
  hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_units[0])])

  # Add a variable number of more hidden layers
  layer_sizes = zip(hidden_units[:-1], hidden_units[1:])
  hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

  net = OrderedDict()

  for i in range(len(hidden_layers)):
    net.update({'fc{}'.format(i): hidden_layers[i]})

    if i+1 < len(hidden_layers):
      net.update({'relu{}'.format(i): nn.ReLU()})
      if drop_p:
        net.update({'dropout{}'.format(i): nn.Dropout(p=drop_p)})

  # optimizer should be NLLLoss
  net.update({'output': nn.LogSoftmax(dim=1)})

  return nn.Sequential(net)

def validation(device, model, dataloaders, data_type, by=1):
  print('Starting validation.. [%s]' % data_type)
  correct = 0
  total = 0
  model.to(device)
  model.eval()
  with torch.no_grad():
    for ii, (images, labels) in enumerate(dataloaders[data_type]):
      if (ii % by) == 0:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(str(correct) +'/'+ str(total))

    print('Accuracy of the network on the validation images: %d %%' % (100 * correct / total))


def training(device, model, dataloaders, image_datasets, criterion, optimizer, epochs=3, print_every=40, save_dir='save_dir.pth'):
    
  def get_lr(optimizer):
    for param_group in optimizer.param_groups:
       return param_group['lr']
    
  steps = 0
  model.to(device)
  model.train()
  print('Starting training.. ')
  for e in range(epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(dataloaders['training']):
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()

      # Forward and backward passes
      outputs = model.forward(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      steps += 1

      if steps % print_every == 0:
        print("Epoch: {}/{}... ".format(e+1, epochs),
              "Loss: {:.4f}".format(running_loss/print_every))
        
        """The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs"""
        save_dict = {
            'model': model.state_dict(),
            'classifier': model.classifier,
            'arch': model.__class__.__name__,
            'class_to_idx': image_datasets['training'].class_to_idx,
            'learning_rate': get_lr(optimizer),
            'epochs' : e
        }
        
        torch.save(save_dict, save_dir)
        running_loss = 0

    validation(device, model, dataloaders, 'validation', by=10)
    model.train()

def get_savedir(save_dir, arch, hidden_units):

  def get_savename(arch, hidden_units):
    save_name = arch
    if hidden_units:
      for unit in hidden_units:
        save_name += '_%d' % unit
    else:
      save_name += '_none_hidden_layer'
    save_name += '.pth'
    return save_name

  if save_dir:
    return save_dir
  else:
    path = os.getcwd()
    file_name = get_savename(arch, hidden_units)
    return os.path.join(path, file_name)

def is_exist_savefile(save_dir):
  return os.path.isfile(save_dir)



def main():
  args = parse_args()

  data_dir = args.data_dir
  save_dir = args.save_dir
  gpu = args.gpu
  arch = args.arch
  learning_rate = args.learning_rate
  hidden_units = args.hidden_units
  epochs = args.epochs
  dropout = args.dropout
  full_model_update = args.full_model_update
  save_dir = get_savedir(save_dir, arch, hidden_units)

  print('='*10+'Params'+'='*10)
  print('Data dir:          {}'.format(data_dir))
  print('Arch:              {}'.format(arch))
  print('Hidden units:      {}'.format(hidden_units))
  print('Learning rate:     {}'.format(learning_rate))
  print('Epochs:            {}'.format(epochs))
  print('GPU:               {}'.format(gpu))
  print('Save file:         {}'.format(save_dir))
  print('Full Model Update: {}'.format(full_model_update))

  train_dir = data_dir + '/train'
  valid_dir = data_dir + '/valid'
  test_dir = data_dir + '/test'

  device = torch.device('cuda' if gpu else 'cpu')
  means = [0.485, 0.456, 0.406]
  deviations  = [0.229, 0.224, 0.225]

  # TODO: Define your transforms for the training, validation, and testing sets
  data_transforms = {
    'training': transforms.Compose([transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means, deviations)
                                    ]),

    'validation': transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(means, deviations)
                                      ]),

    'testing': transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(means, deviations)
                                   ])
  }

  # TODO: Load the datasets with ImageFolder
  image_datasets = {
    'training': datasets.ImageFolder(train_dir, transform=data_transforms['training']),
    'validation': datasets.ImageFolder(valid_dir, transform=data_transforms['validation']),
    'testing': datasets.ImageFolder(train_dir, transform=data_transforms['testing'])
  }

  # TODO: Using the image datasets and the trainforms, define the dataloaders
  dataloaders = {
    'training': torch.utils.data.DataLoader(image_datasets["training"], batch_size=32, shuffle=True),
    'validation': torch.utils.data.DataLoader(image_datasets["validation"], batch_size=32, shuffle=True),
    'testing': torch.utils.data.DataLoader(image_datasets["testing"], batch_size=32)
  }

  model, input_size = get_architecture(arch)
  classifier = get_classifier(input_size, hidden_units, 102, drop_p=dropout)
  model.classifier = classifier

  if is_exist_savefile(save_dir):
    print('Existing save file detected, now loading %s...' % save_dir)
    load = torch.load(save_dir)
    print(load['arch'])
    model.load_state_dict(load['model'])
    print('NN model successfully loaded..')
  else:
    print('Existing save file not found, creating save file at %s...' % save_dir)

  print(model)
    
  criterion = nn.NLLLoss()
  if full_model_update:
    for param in model.parameters():
      param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print('Training Full NN.. ')
  else:
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    print('Training only classification.. add option --full_model_update True to change Full Training NN mode..')
    
  training(device, model, dataloaders, image_datasets, criterion, optimizer, epochs=epochs, save_dir=save_dir)
  print('Just finished training network.. testing will begin soon..')
  validation(device, model, dataloaders, 'testing', by=5)

if __name__ == '__main__':
  main()
# python train.py ./flowers --arch densenet --learning_rate 0.0001 --hidden_units 512 512 --epochs 3 --gpu
