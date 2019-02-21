# -*- coding: utf-8 -*-
import os
import json
import ssl

try:
    from urllib.request import urlopen
except:
    from urllib2 import urlopen
import time

filtersize = 200000

def get_img_by_url(data_dir, name, url):
    try:
        # req = urllib2.Request(url, headers = header)
        context = ssl._create_unverified_context()
        raw_img = urlopen(url, context = context).read()
        filesize = raw_img.__sizeof__()
        with open(data_dir+'/'+name+'.jpg', 'wb') as f:
            f.write(raw_img)
        return filesize , raw_img

    except Exception as e:
        print (e)
        return False , e

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
