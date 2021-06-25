import csv
import os
import cv2
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

# lot imports
from shapely.geometry import box, Polygon

# infrastructure imports
import requests
from datetime import date, datetime
import json
import ftplib
from ast import literal_eval


flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('update_server', False, 'send results to server')
flags.DEFINE_boolean('fps', False, 'calculate and print fps to console')
flags.DEFINE_boolean('save_img', False, 'save analysed img to local direcory')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')

flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')

# used RGB colors
red = (212, 14, 58)
green = (15, 212, 15)
blue = (15, 101, 212)
black = (0, 0, 0)

# load data.json with all information about each occupied parking space
spots = []  # all spots with all information from data.json
lots = []  # all lots with sub-arrays for each spot: lots[i]

with open('data/lots/data.json', 'r') as f:
    spots = json.load(f)

for i in spots:
    l = i["lots"]
    l = literal_eval(l)
    lots.append(l)
    #print(l)


# send taken lots to webserver which saves them in json file
def update_json(spot_id, taken_lots, finalheatmap, utime):
    url = 'https://sparkle-network.com/app/process_data.php'

    lot_ids = taken_lots['lot_ids']
    car_ids = taken_lots['car_ids']
    timespans = taken_lots['timespan']

    taken_lots = []
    for i in range(len(lot_ids)):
        tmp = {'lot_id': lot_ids[i], 'car_id': car_ids[i], 'timespan': timespans[i]}
        taken_lots.append(tmp)
    #taken_lots = json.dumps(taken_lots)

    #obj = {'spot_id': spot_id, 'taken_lots': taken_lots, 'finalheatmap': finalheatmap, 'utime': utime}
    obj = {'taken_lots': taken_lots, 'finalheatmap': finalheatmap, 'utime': utime}
    obj = json.dumps(obj)
    update = {'spot_id': spot_id, 'obj': obj}
    resp = requests.post(url, params=update)

    #print("send to server:\n{}\n".format(update))



def write_json(data, filename):
    path = "data/data_log/server_data/"
    with open(path + filename,'w') as f:
        json.dump(data, f, indent=4)
        print("Successfully wrote data to", filename)
        f.close()
    f = open(path + filename, 'rb')
    session = SESSION  # add credentials here
    session.storbinary('STOR data/{}'.format(filename), f)
    session.quit()


def update_server(filename, senddata):
    path = "data/data_log/server_data/"
    with open(path + filename) as json_file:
        data = json.load(json_file)

        if(filename == "total_usage.json"):
            # define day for total usage entry --> section in json file
            today = date.today()
            datelog = today.strftime("%d.%m.%Y")
            entry = senddata


        if(filename == "daily_usage.json"):
            # define datelog for current day --> section in json file
            today = date.today()
            datelog = "d" + today.strftime("%d%m%Y")
            # log time
            now = datetime.now()
            now = now.strftime("%H:%M")  # formats time for logtime to 00:00

            try:
                entry = data[datelog]
                line = {now: senddata}
                entry.update(line)
            except:
                entry = {now: senddata}


        if(filename == "heatmap.json"):
            # define datelog for current day --> section in json file
            today = date.today()
            datelog = today.strftime("%d.%m.%Y")

            try:
                entry = np.array(data[datelog])
                senddata = np.array(senddata)
                entry = np.add(entry, senddata)
                entry = entry.tolist()
            except:
                entry = senddata


        new_line = {datelog: entry}  # e.g {'00:00': 25} // {'02.10.2020': 40}
        data.update(new_line)  # append dict object with new data
        write_json(data, filename)  # append data in json file

        json_file.close()



# write raw data of each analysis round to local csv file (for later analysis/ as backup)
def append_csv_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        # create write object from csv module
        csv_writer = csv.writer(write_obj)
        csv_writer.writerow(list_of_elem)


def save_analysed_img(result):
    folder = datetime.now().strftime("%d_%m_%y")
    path = os.getcwd() + "/data/data_log/analysed_images/" + folder
    if not os.path.exists(path):
        os.makedirs(path)
    filename = datetime.now().strftime("%H_%M_%S") + ".jpg"
    path = os.path.join(path, filename)
    cv2.imwrite(path, result)
    print("saved img")






def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize deepsort tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = 416

    # load model
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    out = None

    ### csv config ###
    a = 30
    calc_stop = 0
    filename = datetime.now().strftime('%d_%m_%y')
    filename = 'data/data_log/csv_log/{}.csv'.format(filename)

    ### heatmap config ###
    heatmap = []  # heatmap which gets incremented over the day
    for i in range(len(lots)):
        numlot = [0] * len(lots[i])
        heatmap.append(numlot)

    ### server data config ###
    unique_cars = []

    ### intervall config ###
    imageProcessingInterval = 20  # default imageProcessing interval in seconds
    lastProcessed = time.time()  # last processed image
    last_round = []  # initialize last_round which stores last round taken_lot config

    # while video is running
    while True:
        # processes frame only in given time interval
        if lastProcessed + imageProcessingInterval < time.time():
            try:
                lastProcessed = time.time()

                frames = []

                vid = cv2.VideoCapture("http://96.56.250.139:8200/mjpg/video.mjpg")

                if not vid.isOpened():
                    raise IOError("We cannot open webcam", spots[s])

                return_value, frame = vid.read()
                if return_value:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                else:
                    print('Video has ended or failed, try a different video format!')
                    # since the whole loop is in a try-block loop doesnt need to break
                    # which makes the script more robust for temporary camera or internet connection errors
                    #break

                # crop all camera frames to same width to add them to one window
                y = spots[0]["crop"]['y']  # todo: replace '3' with 'i' for data.json loop
                x = spots[0]["crop"]['x']
                h = 600
                w = 600
                frame = frame[y:y + h, x:x + w]

                frame_size = frame.shape[:2]
                image_data = cv2.resize(frame, (input_size, input_size))
                image_data = image_data / 255.
                image_data = image_data[np.newaxis, ...].astype(np.float32)
                start_time = time.time()

                #
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
                scores = scores.numpy()[0]
                scores = scores[0:int(num_objects)]
                classes = classes.numpy()[0]
                classes = classes[0:int(num_objects)]

                # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
                original_h, original_w, _ = frame.shape
                bboxes = utils.format_boxes(bboxes, original_h, original_w)

                # store all predictions in one parameter for simplicity when calling functions
                pred_bbox = [bboxes, scores, classes, num_objects]

                # read in all class names from config
                class_names = utils.read_class_names(cfg.YOLO.CLASSES)

                # by default allow all classes in .names file
                allowed_classes = list(class_names.values())

                # loop through objects and use class index to get class name
                names = []
                for i in range(num_objects):
                    class_indx = int(classes[i])
                    class_name = class_names[class_indx]
                    names.append(class_name)
                names = np.array(names)

                # encode yolo detections and feed to tracker
                features = encoder(frame, bboxes)
                detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

                # initialize color map
                cmap = plt.get_cmap('tab20b')
                colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

                # run non-maxima supression
                boxs = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                classes = np.array([d.class_name for d in detections])
                indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]

                # call the tracker
                tracker.predict()
                tracker.update(detections)

                # update tracks
                car_boxes = []  # cars as box polygon
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()
                    class_name = track.get_class()

                    # draw bbox with tracking id on screen
                    color = colors[int(track.track_id) % len(colors)]
                    color = [i * 255 for i in color]
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.5, (255,255,255), 2)

                    xmin, ymin, xmax, ymax, car_id = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), track.track_id
                    car = box(xmin, ymin, xmax, ymax)
                    car_boxes.append([car, car_id])


                car_boxes = np.array(car_boxes)
                taken_lots = {  # dict stores all taken lots of current analysis round
                    'lot_ids': [],
                    'car_ids': [],
                    'timespan': []
                }

                print("\n######### new round ##############")

                s = 0  # todo: add data.json-loop to loop through multiple cameras in one analysing session
                for i in range(len(lots[s])):  # loop through all lots of parking space
                    # set each parking lot to 'free' as default and change if is blocked
                    lines = green
                    free = '1'
                    cords = np.array(lots[s][i]['cords'], np.int32)  # format coordinates to numpy array with 32-bits ints
                    x1, y1 = cords[0]
                    l = Polygon(cords)
                    lot_id = lots[s][i]['id']

                    for j in range(len(car_boxes)):  # loop through all cars
                        c = car_boxes[j][0]  # each detected car
                        car_id = car_boxes[j][1]  # each detected car
                        intsec = c.intersection(l).area  # intersection of each car with this lot
                        # print(round(int))
                        r = round(intsec / l.area, 2)
                        # print(f" car-{j}: {r}")

                        ### lot is blocked ###
                        if r > .5:  # minimum percentage of how much of the lot neeeds to be blocked
                            lines = red
                            free = '0'

                            # increment heatmap index for parking-lot i
                            heatmap[s][i] += 1
                            # add car id, if new/unique to count unique visitors of day
                            if not car_id in unique_cars:
                                unique_cars.append(car_id)

                            # save taken lot information in taken_lot dict
                            taken_lots['lot_ids'].append(lot_id)
                            taken_lots['car_ids'].append(car_id)
                            # print("\nlot_id", lot_id)


                            # check if lot is still blocked by same car
                            try:
                                # car was blocking this lot already in last round
                                lot_index = last_round['lot_ids'].index(str(lot_id))  # searches for current lot in last_round
                                if car_id == last_round['car_ids'][lot_index]:
                                    # lot is still blocked by same car
                                    timespan = last_round['timespan'][lot_index]
                                    # print("still blocked by car", car_id)
                                else:
                                    # lot is blocked by new car (id)
                                    now = datetime.now()
                                    now = now.strftime('%H:%M')
                                    timespan = now  # saving blocking_time to current time
                                    # print("now blocked by car", car_id)


                            except:
                                # new car is blocking this lot
                                print("new car - lot_id", lot_id)
                                now = datetime.now()
                                now = now.strftime('%H:%M')
                                timespan = now  # saving blocking_time to current time
                                # print("now blocked by car", car_id)

                            finally:
                                # update array with appropriate timespan
                                taken_lots['timespan'].append(timespan)


                            ### finally ###
                            #taken_lots.append(lot_id)
                            break  # break car loop for this lot, bc it is blocked by a car

                    cv2.drawContours(frame, [cords], -1, lines, 3)  # draw lot in correct color
                    cv2.putText(frame, lot_id, (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, .6, black, 1)


                #### update databases ####
                # print last updated timestamp on camerascreen
                now = datetime.now().strftime('%H:%M:%S')
                cv2.putText(frame, now, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, .6, red, 2)
                if FLAGS.update_server:
                    print("Update Server:", now)
                    # update_json(spots[s]["id"], taken_lots, heatmap[s], now)
                    # update_server("daily_usage.json", len(taken_lots['lot_ids']))
                    # update_server("total_usage.json", len(unique_cars))
                    # update_server("heatmap.json", heatmap[s])

                heatmap[s] = [0] * len(lots[s])  # reset heatmap array

                # write round to csv --> if its before 00:00
                if calc_stop > time.time():
                    print("Update CSV:", now)

                    # create list for csv
                    lot_ids = taken_lots['lot_ids']
                    car_ids = taken_lots['car_ids']
                    csv_row = [datetime.now().strftime('%H:%M')]

                    for i in range(len(lots[s])):
                        try:
                            ind = lot_ids.index(str(i))
                            csv_row.append(car_ids[ind])
                            # print("added")

                        except ValueError:
                            csv_row.append(0)
                            # print("not inside")

                    # Append a list as new line to an old csv file
                    append_csv_as_row(filename, csv_row)

                # create new csv and start round for day
                else:
                    unique_cars = []  # reset unique car count on midnight

                    print("reset Heatmap")

                    FMT = '%H:%M:%S'

                    a += 10 #debug: create new csv every x minutes
                    release_time = '00:00:00'.format(a)  # end time
                    #release_time = '19:00:00'.format(a)  # end time
                    release_time = datetime.strptime(release_time, FMT)

                    now = datetime.now().strftime("%H:%M:%S")
                    now = datetime.strptime(now, FMT)

                    tdelta = release_time - now
                    calc_stop = time.time() + tdelta.seconds
                    #print("seconds until calc_stop:", tdelta.seconds)

                    print("create new CSV")
                    filename = datetime.now().strftime('%d_%m_%y')
                    filename = 'data/data_log/csv_log/{}.csv'.format(filename)
                    print(filename)

                    with open(filename, 'a+', newline='') as write_obj:
                        # Create a writer object from csv module
                        list_of_elem = ['timestamp']
                        for i in range(len(lots[s])):
                            list_of_elem.append("lot{}".format(i))
                        csv_writer = csv.writer(write_obj)
                        # Add contents of list as last row in the csv file
                        csv_writer.writerow(list_of_elem)
                        write_obj.close()


                last_round = taken_lots  # save taken_lots of round in "last" round

                # add analysed frame to window of all analysed cameras from data.json
                frames.append(frame)

                if FLAGS.fps:
                    # calculate frames per second of running detections
                    fps = 1.0 / (time.time() - start_time)
                    print("FPS: %.2f" % fps)

                result = np.asarray(frame)
                result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if not FLAGS.dont_show:
                    cv2.imshow("Parking Lot Video", result)

                if FLAGS.save_img:  # save analysed img to local directory for manual evaluation of accuracy
                    save_analysed_img(result)


            except Exception as err:
                print("----- Couldn't process frame ----")
                print(err)
                
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
