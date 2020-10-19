import os
import cv2
import json
import pandas as pd
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
from shapely.geometry import box, Polygon
from ast import literal_eval
from sklearn.metrics import accuracy_score, confusion_matrix


def format_boxes(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        box[0], box[1], box[2], box[3] = xmin, ymin, xmax, ymax
    return bboxes


def get_labels(img_path):
    input_size = 416
    lots = []
    predicted_labels = []
    true_labels = []

    with open('data/lots/data.json', 'r') as f:
        spots = json.load(f)

    for i in spots:
        l = i["lots"]
        l = literal_eval(l)
        lots.append(l)

    FLAGS = tf.compat.v1.flags.FLAGS
    tf.compat.v1.flags.DEFINE_integer('size', 416, 'resize images to')
    tf.compat.v1.flags.DEFINE_float('iou', 0.45, 'iou threshold')
    tf.compat.v1.flags.DEFINE_float('score', 0.50, 'score threshold')
    tf.compat.v1.flags.DEFINE_boolean('update_server', False, 'send results to server')
    tf.compat.v1.flags.DEFINE_boolean('fps', False, 'calculate and print fps to console')
    tf.compat.v1.flags.DEFINE_boolean('save_img', False, 'save analysed img to local direcory')
    tf.compat.v1.flags.DEFINE_boolean('dont_show', False, 'dont show video output')
    tf.compat.v1.flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
    tf.compat.v1.flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
    tf.compat.v1.flags.DEFINE_string('output', None, 'path to output video')
    tf.compat.v1.flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
    tf.compat.v1.flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
    tf.compat.v1.flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')

    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    path_to_annots = os.path.join(img_path, '_annotations.csv')
    df = pd.read_csv(path_to_annots).drop(columns=['width', 'height', 'class'])

    for name in sorted(os.listdir(img_path)):
        if name[-4:] == '.jpg':
            true_boxes = df[df.filename == name].reset_index(drop=True)
            path_to_img = os.path.join(img_path, name)
            frame = cv2.imread(path_to_img)
            #image = cv2.imread(path_to_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=FLAGS.iou,
                score_threshold=FLAGS.score
            )

            # convert data to numpy arrays and slice out unused elements
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = format_boxes(bboxes, original_h, original_w)

            s = 3
            for i in range(len(lots[s])):  # loop through all lots of parking space
                free = 1
                free_true = 1
                cords = np.array(lots[s][i]['cords'], np.int32)  # format coordinates to numpy array with 32-bits ints
                cords[:, 0] = cords[:, 0] * (416/1280)
                cords[:, 1] = ((cords[:, 1]) + 230) * (416/720)
                l = Polygon(cords)
                #cv2.drawContours(image, [cords], -1, (0, 255, 255), 1)

                for j in range(len(bboxes)):  # loop through all cars
                    xmin, ymin, xmax, ymax = int(bboxes[j][0]), int(bboxes[j][1]), int(bboxes[j][2]), int(bboxes[j][3])
                    car = box(xmin, ymin, xmax, ymax)
                    intsec = car.intersection(l).area  # intersection of each car with this lot
                    r = round(intsec / l.area, 2)
                    #cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 200, 20), 2)

                    if r > .5:  # minimum percentage of how much of the lot neeeds to be blocked
                        free = 0
                        break

                if free == 0:
                    predicted_labels.append(0)
                else:
                    predicted_labels.append(1)

                for j in range(len(true_boxes)):  # loop through all cars
                    xmin_true, ymin_true, xmax_true, ymax_true = int(true_boxes.loc[j, 'xmin']), int(
                        true_boxes.loc[j, 'ymin']), int(true_boxes.loc[j, 'xmax']), int(true_boxes.loc[j, 'ymax'])
                    boxes_true = [[xmin_true, ymin_true, xmax_true, ymax_true]]
                    car_true = box(boxes_true[0][0], boxes_true[0][1], boxes_true[0][2], boxes_true[0][3])
                    intsec_true = car_true.intersection(l).area  # intersection of each car with this lot
                    r_true = round(intsec_true / l.area, 2)
                    #cv2.rectangle(image, (xmin_true, ymin_true), (xmax_true, ymax_true), (0, 20, 200), 3)

                    if r_true > .5:  # minimum percentage of how much of the lot neeeds to be blocked
                        free_true = 0
                        break

                if free_true == 0:
                    true_labels.append(0)
                else:
                    true_labels.append(1)

            #cv2.imshow('Window', image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

    return predicted_labels, true_labels


if __name__ == '__main__':
    path_to_imgs = './training_files/test'
    predicted, labels = get_labels(path_to_imgs)
    print(f"Accuracy of the predictions: {accuracy_score(labels, predicted)}")
    print(f"Confusion Matrix:\n {confusion_matrix(labels, predicted)}")
