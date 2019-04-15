# if using conda, conda install -c conda-forge opencv
import cv2
from xml.dom import minidom
import re
import os
import numpy as np

def svg_to_data(folder, lst):
    train = open(folder+"/task/train.txt").read().splitlines()
    valid = open(folder+"/task/valid.txt").read().splitlines()
    train_path = "data/train"
    valid_path = "data/valid"
    for img in lst:
        print("currently working on file "+img)
        dest_path = ''
        if img in train:
            dest_path = train_path
        if img in valid:
            dest_path = valid_path
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        doc = minidom.parse(folder + "/ground-truth/locations/"+img+".svg")  # parseString also exists
        path_strings = [path.getAttribute('d') for path
                        in doc.getElementsByTagName('path')]
        ids_strings = [path.getAttribute('id') for path
                        in doc.getElementsByTagName('path')]
        doc.unlink()
        i = 0
        for path in path_strings:
            wordList = re.sub(" ", " ",  path).split()
            x = list(map(float,wordList[2::3]))
            y = list(map(float,wordList[1::3]))
            x = [round(el) for el in x]
            y = [round(el) for el in y]
            pts = []
            j = 0
            for el in x:
                pts.append([y[j],el])
                j = j + 1
            pts = np.array(pts)
            img2 = cv2.imread(folder + "/images/"+img+".jpg",0)
            ret, imgf = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            ## (1) Crop the bounding rect
            rect = cv2.boundingRect(pts)
            x,y,w,h = rect
            croped = imgf[y:y+h, x:x+w].copy()

            ## (2) make mask
            pts = pts - pts.min(axis=0)

            mask = np.zeros(croped.shape[:2], np.uint8)
            cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

            ## (3) do bit-op
            dst = cv2.bitwise_and(croped, croped, mask=mask)

            ## (4) add the white background
            bg = np.ones_like(croped, np.uint8)*255
            cv2.bitwise_not(bg,bg, mask=mask)
            dst2 = bg+ dst
            resized_image = cv2.resize(dst2, (150, 75)) 
            cv2.imwrite(os.path.join(dest_path , ids_strings[i]+'.jpg'), resized_image)
            i = i + 1