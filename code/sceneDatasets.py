from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def prepare_data(data):
    imgs, captions, captions_lens, class_ids, keys = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    return [real_imgs, captions, sorted_cap_lens,
            class_ids, keys]


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)
    ret = []
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:        
        for i in range(cfg.TREE.BRANCH_NUM):
            # print(imsize[i])            
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Resize(imsize[i])(img)
            else:
                re_img = img
            ret.append(normalize(re_img))
    
    return ret


class TextDataset(data.Dataset):
    def __init__(self, data_dir='../data/OpenImage', split='train', base_size=64, transform=None):
        # default= '../data/OpenImage'
        self.data_dir = data_dir
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform = transform

        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE
        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split)
        self.sentences = self.load_sentences(data_dir, split)
        self.class_id = self.load_classid(data_dir, split)
        self.number_example = len(self.filenames)

    def load_text_data(self, data_dir, split):
        with open(data_dir+'/'+split+'/captions.pickle', 'rb') as f:
            captions = pickle.load(f)
        with open(data_dir+'/'+split+'/filenames.pickle', 'rb') as f:
            filenames = pickle.load(f)
        
        if split == 'train':
            captions_ = captions[0]
        elif split == 'test':
            # tmp to captions[0] use train data
            captions_ = captions[0]
        ixtoword = captions[2]
        wordtoix = captions[3]
        n_words = len(wordtoix)
        return filenames, captions_, ixtoword, wordtoix, n_words
    
    def load_sentences(self, data_dir, split):
        with open(data_dir+'/'+split+'/sentences.pickle', 'rb') as f:
            sentences = pickle.load(f)
        return sentences
    def load_classid(self, data_dir, split):
        if os.path.isfile(data_dir+'/'+split+'/class_ids.pickle'):
            with open(data_dir+'/'+split+'/class_ids.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(len(self.filenames))
        return class_id
    
    def get_caption(self, index):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[index]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def __getitem__(self, index):
        cls_id = self.class_id[index]
        key = self.filenames[index]
        if self.bbox is not None:
            # bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        #
        img_name = self.filenames[index]
        #t0 = time.time()
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, self.norm)        
        # random select a sentence        
        caps, cap_len = self.get_caption(index)
        return imgs, caps, cap_len, cls_id, key


    def __len__(self):
        return len(self.filenames)
