from config import *
import xml.etree.ElementTree as ET
import cv2
import os
import matplotlib.pyplot as plt
for num in range(410):
    # num = 10
    if (os.path.exists(image_path+'/BloodImage_00%03d.jpg'%num))*\
            (os.path.exists(tree_path+'/BloodImage_00%03d.xml'%num))==False:
        continue
    image = cv2.imread(image_path+'/BloodImage_00%03d.jpg'%num)
    tree = ET.parse(tree_path+'/BloodImage_00%03d.xml'%num)
    # try:
    #     image.shape
    #     print("Checked for shape. Shape is {}".format(image.shape))
    # except AttributeError:
    #     print("Error: Invalid shape.")
    for elem in tree.iter():
        if 'object' in elem.tag or 'part' in elem.tag:
            for attr in list(elem):
                if 'name' in attr.tag:
                    name = attr.text
                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            xmin = int(round(float(dim.text)))
                        if 'ymin' in dim.tag:
                            ymin = int(round(float(dim.text)))
                        if 'xmax' in dim.tag:
                            xmax = int(round(float(dim.text)))
                        if 'ymax' in dim.tag:
                            ymax = int(round(float(dim.text)))
                    if name[0] == "R":
                        cv2.rectangle(image, (xmin, ymin),
                                    (xmax, ymax), (0, 255, 0), 1)
                        cv2.putText(image, name, (xmin + 10, ymin + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (0, 255, 0), 1)
                    if name[0] == "W":
                        print(1)
                        cv2.rectangle(image, (xmin, ymin),
                                    (xmax, ymax), (0, 0, 255), 1)
                        cv2.putText(image, name, (xmin + 10, ymin + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (0, 0, 255), 1)
                    if name[0] == "P":
                        print(2)
                        cv2.rectangle(image, (xmin, ymin),
                                    (xmax, ymax), (255, 0, 0), 1)
                        cv2.putText(image, name, (xmin + 10, ymin + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (255, 0, 0), 1)
    plt.figure(figsize=(16,16))
    plt.imshow(image)
    plt.show()