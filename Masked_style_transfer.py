#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import torchvision.transforms as transforms
import torchvision.models as models

import copy


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


# Load and preprocess images
def load_preprocess(mask_path, style_path, content_path, img_nrows):

    style_mask = Image.open(mask_path)
    style_img = Image.open(style_path)
    content_img = Image.open(content_path)

    # Define image sizes
    width, height = content_img.size
    img_ncols = int(width * img_nrows / height)

    loader = transforms.Compose([
                transforms.Resize((img_nrows, img_ncols)),  # scale imported image
                transforms.ToTensor()])  # transform it into a torch tensor

    def preprocess_image(image):
        # fake batch dimension required to fit network's input dimensions
        image = loader(image).unsqueeze(0)
        return image.to(device, torch.float)

    style_mask = preprocess_image(style_mask).sum(dim=1, keepdim=True) / 3
    style_img = preprocess_image(style_img)
    content_img = preprocess_image(content_img)
    
    return style_mask, style_img, content_img

# A function to display images
def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.figure(figsize = (image.size[0]/96,image.size[1]/96))
    plt.axis('off')
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


# In[4]:


# Content Loss
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


# In[5]:


# Style loss (with or without mask)
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b = number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature, style_mask=None):
        super(StyleLoss, self).__init__()
        self.style_mask = style_mask
        if self.style_mask is None:
            self.target = gram_matrix(target_feature).detach()
        else:
            self.target = gram_matrix(target_feature*self.style_mask).detach()

    def forward(self, input):
        if self.style_mask is None:
            G = gram_matrix(input)
        else:
            G = gram_matrix(input*self.style_mask)
        self.loss = F.mse_loss(G, self.target)
        return input


# In[6]:


# Import the VGG module
cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# In[7]:


# Define the tuned  models
# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(mode, cnn, normalization_mean, normalization_std,
                               style_img, content_img, style_mask,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)
    mask_model = nn.Sequential()

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)
        if name[:2] == "po":
            mask_model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()

            if mode == "POOL":
                output_mask = mask_model(style_mask).expand_as(target_feature)
                assert target_feature.shape == output_mask.shape
                style_loss = StyleLoss(target_feature, output_mask)
            else:
                assert mode == "NAIVE"
                style_loss = StyleLoss(target_feature)

            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    del cnn, normalization, mask_model
    torch.cuda.empty_cache()
            
    model = model[:(i + 1)]

    return model, style_losses, content_losses


# In[8]:


# Define the optimizer
def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


# In[9]:


# Function to run the style transfer
def run_style_transfer(mode, cnn, normalization_mean, normalization_std,
                       content_img, style_img, style_mask, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(mode, cnn,
        normalization_mean, normalization_std, style_img, content_img, style_mask)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image if values are too large
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)
        
    del model, style_losses, content_losses, optimizer
    torch.cuda.empty_cache()

    # a last correction...
    input_img.data.clamp_(0, 1)
    
    if mode == "NAIVE":
        input_img = style_mask * input_img + (1-style_mask) * content_img

    return input_img


# # In[10]:


# style_mask, style_img, content_img = load_preprocess(
#     "./masks/bloomfield-mask.jpg", "./images/forest-colors.jpg", "./images/bloomfield.jpg", 500)

# imshow(style_img, title='Style Image')
# imshow(content_img, title='Content Image')
# imshow(content_img*style_mask.expand_as(content_img), "Masked Input")


# # In[11]:


# # Naive masking
# input_img = content_img.clone()
# output = run_style_transfer("NAIVE", cnn, cnn_normalization_mean, cnn_normalization_std,
#                             content_img, style_img, style_mask, input_img)
# imshow(output)


# # In[12]:


# # Masked style loss
# input_img = content_img.clone()
# output = run_style_transfer("POOL", cnn, cnn_normalization_mean, cnn_normalization_std,
#                             content_img, style_img, style_mask, input_img)
# imshow(output)


# # In[13]:


# del style_mask, style_img, content_img, input_img, output
# torch.cuda.empty_cache()


# # In[14]:


# style_mask, style_img, content_img = load_preprocess(
#     "./masks/la_building_mask.png", "./images/forest-colors.jpg", "./images/la.JPG", 700)

# imshow(style_img, title='Style Image')
# imshow(content_img, title='Content Image')
# imshow(content_img*style_mask.expand_as(content_img), "Masked Input")


# # In[15]:


# # Naive masking
# input_img = content_img.clone()
# output = run_style_transfer("NAIVE", cnn, cnn_normalization_mean, cnn_normalization_std,
#                             content_img, style_img, style_mask, input_img)
# imshow(output)


# # In[16]:


# # Masked style loss
# input_img = content_img.clone()
# output = run_style_transfer("POOL", cnn, cnn_normalization_mean, cnn_normalization_std,
#                             content_img, style_img, style_mask, input_img)
# imshow(output)


# # In[17]:


# del style_mask, style_img, content_img, input_img, output
# torch.cuda.empty_cache()


# # In[18]:


# style_mask, style_img, content_img = load_preprocess(
#     "./masks/sevile-building-mask.jpg", "./images/forest-colors.jpg", "./images/seville.jpg", 650)

# imshow(style_img, title='Style Image')
# imshow(content_img, title='Content Image')
# imshow(content_img*style_mask.expand_as(content_img), "Masked Input")


# # In[19]:


# # Naive masking
# input_img = content_img.clone()
# output = run_style_transfer("NAIVE", cnn, cnn_normalization_mean, cnn_normalization_std,
#                             content_img, style_img, style_mask, input_img)
# imshow(output)


# # In[20]:


# # Masked style loss
# input_img = content_img.clone()
# output = run_style_transfer("POOL", cnn, cnn_normalization_mean, cnn_normalization_std,
#                             content_img, style_img, style_mask, input_img)
# imshow(output)


# # In[21]:


# del style_mask, style_img, content_img, input_img, output
# torch.cuda.empty_cache()


# # In[22]:


style_mask, style_img, content_img = load_preprocess(
    "./masks/la_building_mask.png", "./images/forest-fall-2.jpg", "./images/la.JPG", 700)

# imshow(style_img, title='Style Image')
# imshow(content_img, title='Content Image')
# imshow(content_img*style_mask.expand_as(content_img), "Masked Input")


# # In[23]:


# # Naive masking
# input_img = content_img.clone()
# output = run_style_transfer("NAIVE", cnn, cnn_normalization_mean, cnn_normalization_std,
#                             content_img, style_img, style_mask, input_img)
# imshow(output)


# In[24]:


# Masked style loss
input_img = content_img.clone()
output = run_style_transfer("POOL", cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, style_mask, input_img, num_steps=300,
                            style_weight=1000000, content_weight=1)
imshow(output)
plt.show()


# In[ ]:




