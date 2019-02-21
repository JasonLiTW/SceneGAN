# create date 2019/1/8 NepTuNe
# for cocodataset preprocess annotations
import pickle
import json
import time
import numpy as np
import pandas as pd
import re
import csv
import os
import piexif
from PIL import Image
from collections import defaultdict
from datasets import TextDataset
import threading
import multiprocessing as mp
from multiprocessing import Queue
from miscc.openImage_utils import get_img_by_url



def build_dictionary(train_captions, val_captions, train_id, val_id):
    word_counts = defaultdict(float)
    for key in train_captions.keys():
        for caption in train_captions[key]:
            for word in caption.split(' '):
                print(word)
                word_counts[word] += 1
    for key in val_captions.keys():
        for caption in val_captions[key]:
            for word in caption.split(' '):
                word_counts[word] += 1

    vocab = [w for w in word_counts if word_counts[w] >= 0]

    ixtoword = {}
    ixtoword[0] = '<end>'
    wordtoix = {}
    wordtoix['<end>'] = 0
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    train_captions_new = {}
    for key in train_id:
        for caption in train_captions[key]:
            rev = []
            for w in caption.split(' '):
                if w in wordtoix:
                    rev.append(wordtoix[w])
            if key not in train_captions_new.keys():
                train_captions_new[key] = []
                train_captions_new[key].append(rev)
            else:
                train_captions_new[key].append(rev)

    # for t in train_captions:
    #     rev = []
    #     for w in t:
    #         if w in wordtoix:
    #             rev.append(wordtoix[w])
    #     # rev.append(0)  # do not need '<end>' token
    #     train_captions_new.append(rev)

    val_captions_new = {}
    for key in val_id:
        for caption in val_captions[key]:
            rev = []
            for w in caption.split(' '):
                if w in wordtoix:
                    rev.append(wordtoix[w])
            if key not in val_captions_new.keys():
                val_captions_new[key] = []
                val_captions_new[key].append(rev)
            else:
                val_captions_new[key].append(rev)
    # for t in test_captions:
    #     rev = []
    #     for w in t:
    #         if w in wordtoix:
    #             rev.append(wordtoix[w])
    #     # rev.append(0)  # do not need '<end>' token
    #     test_captions_new.append(rev)

    return [train_captions_new, val_captions_new,
            ixtoword, wordtoix, len(ixtoword)]

def preprocess_dog_cat():
    train_anno_path = '../data/coco/annotations/captions_train2014.json'
    val_anno_path = '../data/coco/annotations/captions_val2014.json'

    dogs = ['dogi', 'hotdogs', 'dog', 'bulldog',
    'sheepdog', 'doggie', 'sheepdogs', 'bulldogs', 'adog',
    'watchdog', 'dogg', 'dogpile', 'dogs', 'lapdog', 'doggy']
    #dogs.sort(key=len, reverse=True)
    cats = ['pussycat', 'bobcat', 'housecat', 'cat', 'cats', 'acat']
    #cats.sort(key=len, reverse=True)
    target_string = dogs+cats
    target_string.sort(key=len, reverse=True)

    signal = ['']
    train_dict = {}
    train_id = []
    val_dict = {}
    val_id = []
    counts = 0
    with open(train_anno_path, 'r') as f:
        train_data = json.load(f)
        for anno in train_data['annotations']:
            for name in target_string:
                if anno['caption'].endswith(name) and 'hot dog' not in anno['caption'] \
                    and 'hotdog' not in anno['caption'] and 'hotdogs' not in anno['caption'] \
                    and 'hot-dogs' not in anno['caption']:
                    if anno['image_id'] not in train_dict.keys():
                        train_dict[anno['image_id']] = []
                        caption = re.sub('[,.!@#$%^&*()_\"\'\n]', '', anno['caption']).lower()
                        # print(caption)
                        train_dict[anno['image_id']].append(caption)
                        train_id.append(anno['image_id'])
                    else:
                        caption = re.sub('[,.!@#$%^&*()_\"\'\n]', '', anno['caption']).lower()
                        # print(caption)
                        train_dict[anno['image_id']].append(caption)
                    counts += 1
                    break
    print("Total train sentences used: %d" % counts)
    counts = 0
    with open(val_anno_path, 'r') as f:
        val_data = json.load(f)
        for anno in val_data['annotations']:
            for name in target_string:
                if anno['caption'].endswith(name) and 'hot dog' not in anno['caption'] \
                    and 'hotdog' not in anno['caption'] and 'hotdogs' not in anno['caption'] \
                    and 'hot-dogs' not in anno['caption']:
                    if anno['image_id'] not in val_dict.keys():
                        val_dict[anno['image_id']] = []
                        caption = re.sub('[,.!@#$%^&*()_\"\'\n]', '', anno['caption']).lower()
                        val_dict[anno['image_id']].append(caption)
                        val_id.append(anno['image_id'])
                    else:
                        caption = re.sub('[,.!@#$%^&*()_\"\'\n]', '', anno['caption']).lower()
                        val_dict[anno['image_id']].append(caption)
                    counts += 1
    print("Total val sentences used: %d" % counts)
    train_captions_new, val_captions_new, ixtoword, wordtoix, length = build_dictionary(train_dict, val_dict, train_id, val_id)
    print("Total train images used: %d" % len(train_captions_new.keys()))
    print("Total val images used: %d" % len(val_captions_new.keys()))
    print(len(ixtoword))
    print(len(wordtoix))
    record_list = [train_captions_new, val_captions_new, ixtoword, wordtoix]
    with open('../data/captions_coco.pickle', 'wb') as f:
        pickle.dump(record_list, f, pickle.HIGHEST_PROTOCOL)

    # create train and val filenames pickle file and each image sentence numbers
    filenames = []
    counts = []
    for image_id in train_id:
        length = 12
        counts.append(len(train_captions_new[image_id]))
        idx = '0'*(length-len(str(image_id)))+str(image_id)
        filenames.append('COCO_train2014_'+idx)
        

    with open('../data/train_filenames.pickle', 'wb') as f:
        pickle.dump(filenames, f, pickle.HIGHEST_PROTOCOL)
    with open('../data/coco/train/dict_ids.pickle', 'wb') as f:
        pickle.dump(train_id, f, pickle.HIGHEST_PROTOCOL)
    with open('../data/coco/train/sentences.pickle', 'wb') as f:
        pickle.dump(counts, f, pickle.HIGHEST_PROTOCOL)

    filenames = []
    counts = []
    for image_id in val_id:
        length = 12
        counts.append(len(val_captions_new[image_id]))
        idx = '0'*(length-len(str(image_id)))+str(image_id)
        filenames.append('COCO_val2014_'+idx)
    with open('../data/val_filenames.pickle', 'wb') as f:
        pickle.dump(filenames, f, pickle.HIGHEST_PROTOCOL)
    with open('../data/coco/test/dict_ids.pickle', 'wb') as f:
        pickle.dump(val_id, f, pickle.HIGHEST_PROTOCOL)
    with open('../data/coco/test/sentences.pickle', 'wb') as f:
        pickle.dump(counts, f, pickle.HIGHEST_PROTOCOL)

def preprocess_openimage(confidence): 
    # preprocess original OpenImage Dataset annotation file

    def job(confidence, start, end, order):
        # use multiple process to accelerate procedure
        print("Thread order: %d" % order)
        print("Work lower: %d, upper: %d" % (start, end))
        class_descriptions = pd.read_csv('../data/OpenImage/class-descriptions.csv', header=None)
        ixtolabel = {}
        for i in range(class_descriptions.shape[0]):
            ixtolabel[class_descriptions.iloc[i][0]] = class_descriptions.iloc[i][1]
        # print("Find Same Labels preprocessing! Strart count time!")
        # t0 = time.clock()
        # dataset = TextDataset('../data/coco', 'train', base_size=299)
        # wordtoix = dataset.wordtoix
        # df_labels = pd.read_csv('../data/OpenImage/class-descriptions.csv', header=None)
        # labels = {}
        # id_labels = {}
        # for i in range(df_labels.shape[0]):
        #     labels[df_labels.iloc[i][1].lower()] = df_labels.iloc[i][0]
        #     id_labels[df_labels.iloc[i][0]] = df_labels.iloc[i][1].lower()
        # #print(labels.keys())
        # same_labels = []
        # counts = 0
        # for key in list(labels.keys()):
        #     if key in wordtoix:
        #         same_labels.append(key)
        #         counts += 1
        # print(counts)
        machine_label_path = 'train-annotations-machine-imagelabels.csv'
        # del df_labels
        df_machine_labels = pd.read_csv('../data/OpenImage/'+machine_label_path)
        image_dict = {}
        # print('Thread order: %d, Finish preprocess same labels, Used time: %.2f' % (order, time.clock()-t0))
        print('\nThread order: %d, Start go through machine-label data! Start count time!' % order)
        t0 = time.time()
        for i in range(start, end):
            if ((i-start) % 400000) == 0 and (i-start) != 0:
                print('Thread order: %d, Now %.2f%% Completed !, Used time: %.2f' % (order, ((i-start)/(end-start))*100., time.time()-t0))
                t0 = time.time()
            label = ixtolabel[df_machine_labels.iloc[i]['LabelName']]            
            if df_machine_labels.iloc[i]['ImageID'] not in image_dict and df_machine_labels.iloc[i]['Confidence'] >= confidence:
                image_dict[df_machine_labels.iloc[i]['ImageID']] = []
                image_dict[df_machine_labels.iloc[i]['ImageID']].append(label)
            elif df_machine_labels.iloc[i]['Confidence'] >= confidence:
                image_dict[df_machine_labels.iloc[i]['ImageID']].append(label)
        print('Thread order: %d, Finsih machine-label, Total Image: %d' % (order, len(list(image_dict.keys()))))
        save_path = '../data/machine-data' + str(order) + '.pickle'
        with open(save_path, 'wb') as f:
            pickle.dump(image_dict, f, pickle.HIGHEST_PROTOCOL)
    def merge(mp_num, confidence):
        filenames = ['machine-data' + str(i+1) + '.pickle' for i in range(mp_num)]
        dict_ = {}
        for index, filename in enumerate(filenames):
            path = '../data/'+filename
            print("Start merge file %d/%d" % (index+1, mp_num))
            with open(path, 'rb') as f:
                data = pickle.load(f)                
                for key in data.keys():
                    if key not in dict_:
                        dict_[key] = []
                        for target in data[key]:
                            dict_[key].append(target)
                    else:
                        for target in data[key]:
                            dict_[key].append(target)
        with open('../data/machine-merge-'+str(confidence)+'.pickle', 'wb') as f:
            pickle.dump(dict_, f, pickle.HIGHEST_PROTOCOL)
    mps = []
    mp_num = 11
    total_num = 78977695
    slides = np.arange(0, total_num, total_num//mp_num)
    for i in range(mp_num):
        mps.append(mp.Process(target=job, args=(confidence, slides[i], slides[i+1], i+1)))
        mps[i].start()
    for j in range(mp_num):
        mps[j].join()
    print("========================Start merge all files========================")
    merge(mp_num, confidence)    
    print("=========================Main thread Waiting=========================")

def get_keyword_imgs(keywords, num_pictures):    
    def job(q, df, num_pictures, order):
        with open('../data/machine-merge.pickle', 'rb') as f:
            data = pickle.load(f)
            keys = list(data.keys())
        while not q.empty():
            word1, word2 = q.get()
            print('Thread %d Start download imgs for word1: %s and word2: %s\nLeave q size: %d' % (order, word1, word2, q.qsize()))
            data_train_dir = '../data/OpenImage/train/' + word1 + '_' + word2
            data_val_dir = '../data/OpenImage/val/' + word1 + '_' + word2
            count = 1
            if not os.path.exists(data_train_dir):
                os.makedirs(data_train_dir)
            if not os.path.exists(data_val_dir):
                os.makedirs(data_val_dir)
            for key in keys:
                if count == num_pictures:
                    break
                if word1 in data[key] and word2 in data[key] and 'people' not in data[key] and \
                    'man' not in data[key] and 'woman' not in data[key]:
                    if count % 100 == 0:
                        print('Thread %d download %d imgs for word1: %s and word2: %s' % (order, count, word1, word2))
                    if np.random.random() > 0.1:
                        url = df[df['ImageID'] == key]['OriginalURL'].values[0]
                        get_img_by_url(data_train_dir, key, url)
                    elif np.random.random() <= 0.1:
                        url = df[df['ImageID'] == key]['OriginalURL'].values[0]
                        get_img_by_url(data_val_dir, key, url)
                    count += 1
                
            print('Thread %d Finished download imgs for word1: %s and word2: %s' % (order, word1, word2))
    def clean_null_img(keywords, split):
        for keyword in keywords:
            data_dir = '../data/OpenImage/' + split + '/' + keyword[0] + '_' + keyword[1]
            filenames = os.listdir(data_dir)
            for filename in filenames:
                file_path = data_dir+'/'+filename
                with open(file_path, 'rb') as f:
                    if f.seek(0,2) < 4096:
                        os.remove(file_path)
    def resize(keywords, split):
        for index, keyword in enumerate(keywords):
            data_dir = '../data/OpenImage/' + split + '/' + keyword[0] + '_' + keyword[1]
            print("Start %s split %d" % (split, index))
            new_dir = data_dir+'/resize/'
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)                
            filenames = os.listdir(data_dir)
            for filename in filenames:
                if filename == 'resize':
                    continue
                file_path = data_dir+'/'+filename                
                piexif.remove(file_path)
                image = Image.open(file_path)
                image = resize_image(image, 256)
                image.save(new_dir+filename)
    # df = pd.read_csv('../data/OpenImage/image_ids_and_rotation.csv')
    # q = Queue()
    # for list_words in keywords:
    #     q.put(list_words)
    # mps = []
    # mp_num = 6    
    # for i in range(mp_num):
    #     mps.append(mp.Process(target=job, args=(q, df, num_pictures, i+1)))
    #     mps[i].start()
    # for j in range(mp_num):
    #     mps[j].join()
    # print("========================Start clear Null Imgs========================")
    # clean_null_img(keywords, 'train')
    # clean_null_img(keywords, 'test')
    resize(keywords, 'train')
    resize(keywords, 'test')
    print("=========================Main thread Finished=========================")
    
def preprocess_sceneImage(split):
    keywords = [['forest', 'morning'], ['forest', 'winter'], ['forest', 'night'], ['forest', 'autumn'], \
                ['canal', 'morning'], ['canal', 'night'], ['canal', 'autumn'], ['canal', 'winter'], ['farm', 'morning'], ['mountain', 'morning'],\
                ['mountain', 'night'], ['mountain', 'cloud'], ['mountain', 'sea'], ['mountain', 'ocean'], ['mountain', 'desert'], \
                ['beach', 'winter'], ['beach', 'night'], ['beach', 'morning'], ['beach', 'sunset'], ['rock', 'field'], \
                ['fog', 'forest'], ['fog', 'morning'], ['road', 'forest'], ['road', 'mountain'], ['road', 'night'], \
                ['road', 'morning'], ['road', 'field']]
    scene_spec = ['sand', 'mountain', 'mountain_path', 'mountain_snowy', 'beach', 'beach_house', 'butte', 'canal_urban', 'canal_natural'\
            'canyon', 'cliff', 'corn_field', 'creek', 'desert_road', 'farm', 'field_cultivated', 'field_road', 'forest_broadleaf', \
            'forest_path', 'forest_road', 'glacier', 'grotto', 'harbor', 'hayfield', 'highway', 'iceberg', 'islet', \
            'lagoon', 'ocean', 'railroad_track', 'rainforest', 'river', 'rock_arch', 'sky', 'snowfield', 'valley', \
            'waterfall', 'coast']

    # f/field/cultivated c/canal/urban f/forest/broadleaf c/canal/natural
    captions = pd.read_csv('../data/OpenImage/target.csv', header=None)
    scene_captions = [spe for spe in scene_spec]
    word_counts = defaultdict(float)
    # vocabulary for OpenImage
    for i in range(captions.shape[0]):
        for word in captions.iloc[i][1].split(' '):
            word_counts[word] += 1    
    # vocabulary for scene dataset
    for caption in scene_captions:
        for word in caption.split('_'):
            word_counts[word] += 1

    vocab = [w for w in word_counts if word_counts[w] >= 0]

    ixtoword = {}
    ixtoword[0] = '<end>'
    wordtoix = {}
    wordtoix['<end>'] = 0
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    captions_new = []
    for i in range(captions.shape[0]):
        rev = []
        for word in captions.iloc[i][1].split(' '):
            rev.append(wordtoix[word])
        captions_new.append(rev)
    for spe in scene_captions:
        rev = []
        for word in spe.split('_'):
            rev.append(wordtoix[word])
        captions_new.append(rev)

    filePaths = []
    fileCaptions = []
    caption_lens = []
    class_ids = []

    for index, keyword in enumerate(keywords):
        filenames = os.listdir('../data/OpenImage/'+split+'/'+keyword[0]+'_'+keyword[1]+'/resize')
        for filename in filenames:
            filePaths.append('../data/OpenImage/'+split+'/'+keyword[0]+'_'+keyword[1]+'/resize/'+filename)
            fileCaptions.append(captions_new[index])
            caption_lens.append(len(captions_new[index]))
            class_ids.append(index)
    offset = len(keywords)
    for index, keyword in enumerate(scene_captions):
        count = 0
        if split == 'train':
            with open('../data/scene/places365_train_standard.txt', 'r') as f:
                data = f.readlines()
                filenames = [spec.strip('\n') for spec in data]
            for filename in filenames:
                if filename.split(' ')[0].split('/')[-2] == keyword and count < 1000:                    
                    filePaths.append('../data/scene/data_large' + filename.split(' ')[0])                    
                    count += 1
                elif filename.split(' ')[0].split('/')[-2] == keyword.split('_')[-1] and count < 1000:
                    filePaths.append('../data/scene/data_large' + filename.split(' ')[0])
                    count += 1
                elif count < 1000:                    
                    continue
                elif count >= 1000:
                    break
                fileCaptions.append(captions_new[index+offset])
                caption_lens.append(len(captions_new[index+offset]))
                class_ids.append(index+offset)
    with open('../data/OpenImage/'+split+'/'+'captions.pickle', 'wb') as f:
        tmp = [fileCaptions, [], ixtoword, wordtoix]
        pickle.dump(tmp, f, pickle.HIGHEST_PROTOCOL)
    with open('../data/OpenImage/'+split+'/'+'filenames.pickle', 'wb') as f:
        pickle.dump(filePaths, f, pickle.HIGHEST_PROTOCOL)
    with open('../data/OpenImage/'+split+'/'+'sentences.pickle', 'wb') as f:
        pickle.dump(caption_lens, f, pickle.HIGHEST_PROTOCOL)
    with open('../data/OpenImage/'+split+'/'+'class_ids.pickle', 'wb') as f:
        pickle.dump(class_ids, f, pickle.HIGHEST_PROTOCOL)
        
def resize_image(image, size=256):
    width, height = image.size
    load_size = int( size * 304 / 256)
    image = image.resize([load_size, load_size], Image.BILINEAR)
    # random crop
    crop_size = 304-size
    x = np.random.randint(0, crop_size-1)
    y = np.random.randint(0, crop_size-1)
    image = image.crop((x, y, x+size, y+size))
    
    return image
if __name__ == '__main__':
    print("====Start Preprocess Dataset Script!====")
    keywords = [['forest', 'morning'], ['forest', 'winter'], ['forest', 'night'], ['forest', 'autumn'], \
                ['canal', 'morning'], ['canal', 'night'], ['canal', 'autumn'], ['canal', 'winter'], ['mountain', 'morning'], ['farm', 'morning'], \
                ['mountain', 'night'], ['mountain', 'cloud'], ['mountain', 'sea'], ['mountain', 'ocean'], ['mountain', 'desert'], \
                ['beach', 'winter'], ['beach', 'night'], ['beach', 'morning'], ['beach', 'sunset'], ['rock', 'field'], \
                ['fog', 'forest'], ['fog', 'morning'], ['road', 'forest'], ['road', 'mountain'], ['road', 'night'], \
                ['road', 'morning'], ['road', 'field']]
    num_pictures = 1000
    # get_keyword_imgs(keywords, num_pictures)
    # preprocess_openimage(1.0)
    preprocess_sceneImage(split='train')