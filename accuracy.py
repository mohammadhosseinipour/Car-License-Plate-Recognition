import os
import matplotlib.pyplot as plt
import cv2
import re

test_count=141
counter=0
for n, image_file in enumerate(os.scandir("test_groundtruth")):
    save_path = os.path.join("test_groundtruth", image_file.name)
    GT = open(save_path, 'r')
    save_path = os.path.join("Tunisian_dataset_predected", image_file.name)
    pre = open(save_path, 'r')

    GT_coords=GT.readlines()
    pre_coords=pre.readlines()
    pre_count=len(pre_coords)
    GT_count=len(GT_coords)
    # print(GT_coords)
    pre_counter=0
    GT_counter=0
    for GT_crd in GT_coords:
        try:
            x1,y1,a1,b1=re.findall(r'\d+', GT_crd)
        except:
            test_count=test_count-1
            continue
        # print(x1,y1,a1,b1)
        x1=int(x1)
        y1=int(y1)
        a1=int(a1)
        b1=int(b1)
        max_IoU=0
        for pre_crd in pre_coords:
            x2,y2,a2,b2=re.findall(r'\d+', pre_crd)
            x2=int(x2)
            y2=int(y2)
            a2=int(a2)
            b2=int(b2)

            area1 = (a1-x1)*(b1-y1);
            area2 = (a2-x2)*(b2-y2);
            xx = max(x1, x2)
            yy = max(y1, y2)
            aa = min(a1, a2)
            bb = min(b1, b2)

            w = max(0, aa - xx)
            h = max(0, bb - yy)

            intersection_area = w*h
            union_area = area1 + area2 - intersection_area

            IoU = intersection_area / union_area
            # print(IoU)
            max_IoU=max(max_IoU,IoU)
            if IoU >0.7:
                pre_counter= pre_counter+1
        if max_IoU >0.7:
            GT_counter= GT_counter+1
    # if GT_counter==GT_count :
    #     counter=counter+1
    if GT_counter==GT_count and pre_counter==pre_count:
        counter=counter+1

accuracy=counter/test_count
print("corupptrd:",141-test_count)
print("true_positive:",counter,"accuracy:",accuracy*100)
