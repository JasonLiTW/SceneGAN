# -*- coding: utf-8 -*-
import os
import csv
import time
import argparse
import pickle
import numpy as np

import multiprocessing as mp
import pandas as pd
from queue import Queue

from miscc.google_img_parser import get_img_by_url

example_text = '''example:

python Main.py --csv_path ./test.csv --output_root  ./data

'''

#Main start mutiple threads to work effective
# create the instance
urls_q = Queue()



# parser = argparse.ArgumentParser( epilog = example_text,
#                                     formatter_class = argparse.RawDescriptionHelpFormatter)
# parser.add_argument('--csv_path', required = True, dest = "csv_path", type = str, help = "the csv file path")
# parser.add_argument('--output_root', required = True, dest = "output_root", type = str, help = "the output root")


# args = parser.parse_args()

# csv_path = args.csv_path
# output_root = args.output_root



def download_specific_images(process_num, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    def job(open_images, lower, upper, order):
        print('Process order: %d Start putting url Job!' % order)
        for idx in np.arange(lower, upper):
            id = open_images.iloc[idx]['ImageID']
            url = open_images.iloc[idx]['OriginalURL']
            urls_q.put([id, url]) 
        counts = 0        
        print('Process order: %d Start Download Job!' % order)
        while not urls_q.empty():
            counts += 1
            id, url = urls_q.get()
            size, rawImg = get_img_by_url(url)
            if size:
                with open(output_dir+'/'+id+'.jpg', 'wb') as f:
                    f.write(rawImg)
            if counts % 500000 == 0:
                print('Process order: %d, Download %d images!' % (order, counts))

    open_images = pd.read_csv('../data/OpenImage/image_ids_and_rotation.csv')    
    # for idx in range(open_images.shape[0]):
    #     if (idx+1) % 1000000 == 0:
    #         print('Put url %d !' % idx)
    #     id = open_images.iloc[idx]['ImageID']
    #     url = open_images.iloc[idx]['OriginalURL']
    #     urls_q.put([id, url])    
    # print("Total size: %d~" % urls_q.qsize())
    processes = []
    all_index = np.arange(0, open_images.shape[0], open_images.shape[0]//process_num)
    for i in range(process_num):
        if i == process_num-1:
            plus = open_images.shape[0]-all_index[-1]
            processes.append(mp.Process(target=job, args=(open_images, all_index[i], all_index[-1]+plus, i+1)))
        else:
            processes.append(mp.Process(target=job, args=(open_images, all_index[i], all_index[i+1], i+1)))
        processes[i].start()

    for i in range(process_num):
        processes[i].join()

    print("=============Main process!=============")
    


def main():
    download_specific_images(11, '../data/OpenImage/images')
    # print ("start parsing ...")
    # # add items to the queue
    # QueryList = csv.reader(open(csv_path))
    # for i, each in enumerate(QueryList):
    #     # tmp = [i]
    #     # tmp.append(each[1].decode('utf-8'))
    #     q.put(each)

    # threads = []
    # for i in range(10): # aka number of threads
    #     threads.append(Thread(target = do_task)) # target is the above function
    #     threads[i].start() # start the thread

    # for j in threads:
    #     j.join()

    # #q.join() # this works in tandom with q.task_done //// Queue.join() 實際上意味著等到隊列為空，再執行別的操作
    #         # essentially q.join() keeps count of the queue size
    #         # and q.done() lowers the count one the item is used
    #         # this also stops from anything after q.join() from
    #         # being actioned.
    # print("Done")


if __name__ == "__main__":
    main()
