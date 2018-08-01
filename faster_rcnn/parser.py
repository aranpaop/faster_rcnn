""" 得到训练数据 """
import cv2
import numpy as np
import os


def get_data(img_path, label_path):
    dirs = os.listdir(label_path)[:1000] # 使用前1000张图片训练
    all_imgs = {}
    classes_count = {}
    class_mapping = {}
    # 所使用的KITTI数据集包含'Pedestrian','Person_sitting','Cyclist','Truck','Car','Misc','Van','Tram','DontCare'九个标签，分为两类
    people = ['Pedestrian', 'Person_sitting', 'Cyclist']
    vehicle = ['Truck', 'Car', 'Van', 'Tram']

    for dir in dirs:
        filename = dir[:-3] + 'png'
        if filename not in all_imgs:
            all_imgs[filename] = {}

        img = cv2.imread(os.path.join(img_path, filename))
        (rows, cols) = img.shape[:2]
        all_imgs[filename]['filepath'] = filename
        all_imgs[filename]['width'] = cols
        all_imgs[filename]['height'] = rows
        all_imgs[filename]['bboxes'] = []
        # 取约1/6的数据作为验证集
        if np.random.randint(0, 6) > 0:
            all_imgs[filename]['imageset'] = 'trainval'
        else:
            all_imgs[filename]['imageset'] = 'test'
        with open(os.path.join(label_path,dir), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_split = line.split()
                if line_split[0] in people:
                    class_name = 'people'
                elif line_split[0] in vehicle:
                    class_name = 'vehicle'
                else:
                    continue

                x1 = float(line_split[4])
                y1 = float(line_split[5])
                x2 = float(line_split[6])
                y2 = float(line_split[7])
                classes_count[class_name] = classes_count.get(class_name, 0) + 1
                if class_name not in class_mapping:
                    class_mapping[class_name] = len(class_mapping)

                all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})
    all_data = []
    for key in all_imgs:
        all_data.append(all_imgs[key])

    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)
    # 随机打乱数据
    np.random.shuffle(all_data)
    return all_data, classes_count, class_mapping
