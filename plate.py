import numpy as np
import pandas as pd
import os
import cv2
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

flag=True
level=0
CARS_PLATES=[]
CARS_PLATES_PIC=[]
sim_plates=[]
nine_plates=[]
plate_pic=[]
labels = open(r'C:\Users\mohammad\Desktop\darknet-master\data_dir\classes.names').read()
# labels = open(r'C:\Users\mohammad\Desktop\darknet-master\data_dir\classes.names').read()
print(labels)
weights_path = r'C:\Users\mohammad\Desktop\darknet-master\bin\lapi.weights'
# weights_path = r'C:\Users\mohammad\Desktop\darknet-master\backup\darknet_yolov3_last.weights'
# configuration_path = r'C:\Users\mohammad\Desktop\darknet-master\cfg\darknet-yolov3.cfg'
configuration_path = r'C:\Users\mohammad\Desktop\darknet-master\cfg\darknet_yolov3.cfg'
video_path=r'C:\Users\mohammad\Desktop\darknet-master\test_vids\test3.mp4'
# image_input = cv2.imread(r'C:\Users\mohammad\Desktop\LPR\darkflow-master\image_dataset\000005.png')

num_frame = cv2.imread(r'C:\Users\mohammad\Desktop\darknet-master\num_frame.PNG')

probability_minimum = 0.5
threshold = 0.3

network = cv2.dnn.readNetFromDarknet(configuration_path, weights_path)
network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
layers_names_all = network.getLayerNames()
# print(layers_names_all)
layers_names_output = [layers_names_all[i[0]-1] for i in network.getUnconnectedOutLayers()]
# print(layers_names_output)

model=None
class_names=["0","1","2","3","4","5","6","7","8","9","A","B","D",
          "Gh","H","J","L","M","N","P","PuV","PwD","Sad","Sin",
          "T","Taxi","V","Y"]

plate_checklist=[[True,1],[True,1],[True,2],[True,1],[True,1],[True,1],[True,1],[True,1]]
model_path="character_recognition_model"





def find_plate(model,image_input,network):
    global CARS_PLATES
    global sim_plates
    global plate_pic
    global flag
    # plt.rcParams['figure.figsize'] = (10.0,10.0)
    # plt.imshow(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
    # plt.show()
    blob = cv2.dnn.blobFromImage(image_input, 1/255.0, (416,416), swapRB=True, crop=False)
    # blob_to_show = blob[0,:,:,:].transpose(1,2,0)
    network.setInput(blob)
    output_from_network = network.forward(layers_names_output)
    # print(len(output_from_network[0]))

    np.random.seed(42)
    colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    bounding_boxes = []
    confidences = []
    class_numbers = []
    h,w = image_input.shape[:2]

    for result in output_from_network:
        for detection in result:
            scores = detection[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]
            if confidence_current > probability_minimum:
                box_current = detection[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current.astype('int')
                x_min = int(x_center-(box_width/2))
                y_min = int(y_center-(box_height/2))
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
    if len(results) > 0:
        plates=[]
        for i in results.flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            colour_box_current = [int(j) for j in colours[class_numbers[i]]]
            # plate = image_input[y_min-3:y_min+box_height+3, x_min-3:x_min+box_width+3].copy()
            plate = image_input[y_min-5:y_min+box_height+5, x_min-7:x_min+box_width+7].copy()
            plate_num=character_recongition(plate,model)
            # plate_num=character_recongition_pytesseract(plate)
            # plate_num=character_recongition_easyocr(plate)
            # plates.append(crop_img)
            # plt.imshow(cv2.cvtColor(plates[0], cv2.COLOR_BGR2RGB))
            # plt.show()
            if sum(c.isdigit() for c in plate_num)==7:
                if len(sim_plates)==0 :
                    sim_plates.append(plate_num)
                    # flag=True
                    plate_pic=plate
                elif ((sim_plates[0][0]==plate_num[0] and sim_plates[0][1]==plate_num[1] and sim_plates[0][2]==plate_num[2]) or
                (sim_plates[0][1]==plate_num[1] and sim_plates[0][2]==plate_num[2] and sim_plates[0][3]==plate_num[3]) or
                (sim_plates[0][2]==plate_num[2] and sim_plates[0][3]==plate_num[3] and sim_plates[0][4]==plate_num[4]) or
                (sim_plates[0][3]==plate_num[3] and sim_plates[0][4]==plate_num[4] and sim_plates[0][5]==plate_num[5]) or
                (sim_plates[0][4]==plate_num[4] and sim_plates[0][5]==plate_num[5] and sim_plates[0][6]==plate_num[6]) or
                (sim_plates[0][5]==plate_num[5] and sim_plates[0][6]==plate_num[6] and sim_plates[0][7]==plate_num[7])):
                    # print("sim:",sim_plates[0][0],"num:",plate_num[0])
                    sim_plates.append(plate_num)
                else:
                    # if not flag:
                    if most_frequent(sim_plates)in CARS_PLATES:
                        plate_pic=[]
                        sim_plates=[]
                        continue
                    CARS_PLATES.append(most_frequent(sim_plates))
                    CARS_PLATES_PIC.append(plate_pic)
                    plate_pic=[]
                    sim_plates=[]
                    # flag=False
                # if plate_num not in CARS_PLATES:
                #     CARS_PLATES.append(plate_num)
                cv2.rectangle(image_input, (x_min, y_min), (x_min + box_width, y_min + box_height),
                              colour_box_current, 5)
                text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])], confidences[i])
                cv2.putText(image_input, text_box_current, (x_min, y_min+box_height + 40), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (128,0,128), 3)
                # cv2.putText(image_input, "LP#:"+plate_num, (x_min-20, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX,
                #             2, (128,0,128), 3)
    return([image_input,bounding_boxes,confidences])


def test_plate():
    counter_=0
    for n, image_file in enumerate(os.scandir("Tunisian_dataset_test")):
        image_input = cv2.imread(image_file.path)
        [out,bounding_boxes,confidences]=find_plate(image_input,network)
        print(image_file.name)
        # plt.rcParams['figure.figsize'] = (10.0,10.0)
        # plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        # plt.show()
        if not os.path.isdir("Tunisian_dataset_predected"):
            os.mkdir("Tunisian_dataset_predected")
        save_path = os.path.join("Tunisian_dataset_predected", image_file.name.replace('jpg', 'txt'))
        GT = open(save_path, 'w')
        print(bounding_boxes)
        for i in range(-1,len(bounding_boxes)):
            if len(bounding_boxes)==0:
                counter_=counter_+1
                GT.write("\n")
            else:
                GT.write(str(bounding_boxes[i][0])+ " " + str(bounding_boxes[i][1])+ " " + str(bounding_boxes[i][0]+bounding_boxes[i][2])+ " " + str(bounding_boxes[i][1]+bounding_boxes[i][3]) + "\n")

    print("counter:",counter_)

def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label

def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1

    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

def character_recongition_model():
    ds=tf.keras.preprocessing.image_dataset_from_directory(
        "Iranis_Dataset_Files_resized",
        labels="inferred",
        label_mode='int',
        class_names=class_names,
        color_mode="rgb",
        batch_size=128,
        image_size=(30, 30),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False)

    ds = ds.map(process)
    ds_train,ds_validation,ds_test=get_dataset_partitions_tf(ds,83844)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(int(83844*0.8))
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    ds_validation = ds_validation.cache()
    ds_validation = ds_validation.prefetch(tf.data.experimental.AUTOTUNE)

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(30, 30, 3)),
      tf.keras.layers.Dense(128,activation='relu'),
      tf.keras.layers.Dense(28)
    ])


    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_validation,
    )
    return model

def character_recongition(plate,model):
    # print(plate.shape)
    # plate_w,plate_h,dim=plate.shape
    # print(plate_w,"XX",plate_h)
    # plt.imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
    # plt.show()
    # time.sleep(5)
    global plate_checklist
    plate_checklist=[[True,1],[True,1],[True,2],[True,1],[True,1],[True,1],[True,1],[True,1]]
    try:
        gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
    except:
        return ""
    gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    # print(thresh)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    im2 = gray.copy()
    plate_num = ""
    # print(len(sorted_contours))
    for cnt in sorted_contours:
        x,y,w,h = cv2.boundingRect(cnt)
        height, width = im2.shape
        # if height of box is not tall enough relative to total height then skip
        if height / float(h) > 8:
            # print("continue1")
            continue

        if w / width > 0.4:
            # print("continue2")
            continue

        if w / width < 0.03:
            # print("continue3")
            continue
        # ratio = h / float(w)
        # # if height to width ratio is less than 1.5 skip
        # if ratio < 1.5: continue
        #
        # # if width is not wide enough relative to total width then skip
        # if width / float(w) > 15: continue

        area = h * w
        # print("x:",x," y:",y," w:",w," h:",h," area:",area)
        # if area is less than 100 pixels skip
        if area < 60:
            # print("continue4")
            continue

        # draw the rectangle
        if h / height < 0.3:
            y_min=int(max(0,y-(h/2)))
            y_max=int(min(height,y+(1.5*h)))
        else:
            y_min=y-10
            y_max=y+h+10
        # print("min",max(0,y-(h/2)))
        # print("max",min(height,y+(1.5*h)))
        rect = cv2.rectangle(im2, (x,y_min), (x+w,y_max), (0,255,0),2)
        # grab character region of image
        roi = thresh[y_min:y_max, x-5:x+w+5]
        # perfrom bitwise not to flip image to black text on white background

        roi = cv2.bitwise_not(roi)
        # perform another blur on character region
        # print("roi",roi)
        try:
            roi = cv2.medianBlur(roi, 5)
        except:
            # print("continue5")
            continue
        ready_input=cv2.cvtColor(roi,cv2.COLOR_GRAY2RGB)
        img_array = keras.preprocessing.image.img_to_array(cv2.resize(ready_input, (30,30), interpolation = cv2.INTER_CUBIC))
        img_array = tf.expand_dims(img_array, 0)
        # plt.imshow(ready_input)
        # plt.show()
        # try:
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        # print(
        #     "This image most likely belongs to {} ."
        #     .format(class_names[np.argmax(score)])
        # )
        text=class_names[np.argmax(score)]
        # print(text)
        if plate_check(text)==True:
            plate_num += text
        # except:
        #     print("failed")
        #     text = None
    # if plate_num != None:
    #     print("License Plate #: ", plate_num)
    return plate_num

def plate_check(text):
    flag=False
    req=None
    num=["0","1","2","3","4","5","6","7","8","9"]
    alpha=["A","B","D","Gh","H","J","L","M","N","P","PuV","PwD","Sad","Sin",
              "T","Taxi","V","Y"]
    plate_count=0
    for tp in plate_checklist:
        if tp[0]==True:
            req=tp[1]
            break
        plate_count=plate_count+1

    if req==1 and text in num:
        flag=True
        plate_checklist[plate_count][0]=False
    elif req==2 and text in alpha:
        flag=True
        plate_checklist[plate_count][0]=False

    if plate_count==3 and text=="0":
        flag=False
    return flag
# test_plate()
def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i

    return num

def video_LPR(video_path):
    global temp
    capture = cv2.VideoCapture(video_path)
    while (capture.isOpened()):
        stime = time.time()
        ret, frame = capture.read()
        if ret:
            [out,bounding_boxes,confidences]=find_plate(model,frame,network)
            imS = cv2.resize(out, (960, 540))
            imS=np.concatenate((imS, num_frame), axis = 1)
            for i in range(len(CARS_PLATES)-1,len(CARS_PLATES)-10,-1):
                if i==-1:
                    break
                if CARS_PLATES[i] in CARS_PLATES[:i]:
                    continue
                if len(CARS_PLATES[i])==0:
                    CARS_PLATES.remove(CARS_PLATES[i])
                # print(CARS_PLATES)
                cv2.putText(imS, CARS_PLATES[i], (962, ((len(CARS_PLATES)-i)*60)-30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (128,0,128), 2)
                p_pic = cv2.resize(CARS_PLATES_PIC[i], (130, 50))
                imS[(((len(CARS_PLATES)-i)*60)-55):(((len(CARS_PLATES)-i)*60)-5),1130:]=p_pic
            cv2.imshow('frame',imS)
            print('FPS {:.1f}'.format(1 / (time.time() - stime)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            cv2.imwrite(r'C:\Users\mohammad\Desktop\darknet-master\video_test_results\result3.png',imS)
            capture.release()
            cv2.destroyAllWindows()
            break

if os.path.exists('character_recognition_model/saved_model.pb'):
    print("using pretrained model...")
    model = keras.models.load_model(model_path)
else:
    model=character_recongition_model()
    model.save(model_path)


video_LPR(video_path)

# [out,bounding_boxes,confidences]=find_plate(model,image_input,network)



# plt.rcParams['figure.figsize'] = (10.0,10.0)
# plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
# plt.show()

print("done")




###### TESSERACT AND EASYOCR

# def character_recongition_pytesseract(plate):
#     gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
#     gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
#     gray = cv2.GaussianBlur(gray, (5,5), 0)
#     gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#     kernel = np.ones((3, 3), np.uint8)
#     gray = cv2.bitwise_not(gray)
#     gray = cv2.erode(gray, kernel)
#     gray = cv2.bitwise_not(gray)
#
#     plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
#     plt.show()
#
#     text = pytesseract.image_to_string(plate, lang='fas')
#     print('text:',text)
#     return text[0]
#
#
#
# def character_recongition_pytesseract(plate):
#     gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
#     gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
#     blur = cv2.GaussianBlur(gray, (5,5), 0)
#     # plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
#     # plt.show()
#     ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
#     # print(thresh)
#     rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
#     dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
#     # plt.imshow(cv2.cvtColor(dilation, cv2.COLOR_BGR2RGB))
#     # plt.show()
#     try:
#         contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     except:
#         ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
#
#     im2 = gray.copy()
#     # cv2.drawContours(gray, sorted_contours, -1, (0,255,0), 1)
#     # plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
#     # plt.show()
#     im2 = gray.copy()
#     # create blank string to hold license plate number
#     plate_num = ""
#     # loop through contours and find individual letters and numbers in license plate
#     # areas=[]
#     # x,y,w,h = cv2.boundingRect(sorted_contours[0])
#     # areas.append([x-2,y-2,x+w+2,y+h+2])
#
#     for cnt in sorted_contours:
#         # sub_region=0
#         x,y,w,h = cv2.boundingRect(cnt)
#         height, width = im2.shape
#
#         # # if height of box is not tall enough relative to total height then skip
#         # if height / float(h) > 6: continue
#         #
#         # ratio = h / float(w)
#         # # if height to width ratio is less than 1.5 skip
#         # if ratio < 1: continue
#         #
#         # if width is not wide enough relative to total width then skip
#         # if width / float(w) > 15: continue
#         #
#         area = h * w
#         # if area is less than 100 pixels skip
#         if area <100: continue
#
#         # for region in areas:
#         #     if x>=region[0] and x+w<region[2] and y>=region[1] and y+h<region[3]:
#         #         sub_region=1
#         # if sub_region==1:
#         #     continue
#         # for region in areas:
#         #     if x<region[0]+10 and y<region[1]+10:
#         #         sub_region=1
#         # if sub_region==1:
#         #     continue
#         # areas.append([x-2,y-2,x+w+2,y+h+2])
#
#         # draw the rectangle
#         rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
#         # grab character region of image
#         roi = thresh[y-5:y+h+5, x-5:x+w+5]
#         # perfrom bitwise not to flip image to black text on white background
#         roi = cv2.bitwise_not(roi)
#         # perform another blur on character region
#         # roi = cv2.medianBlur(roi, 5)
#         # plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
#         # plt.show()
#         try:
#             # text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
#             text = pytesseract.image_to_string(gray, lang='fas')
#             # print(pytesseract. get_tesseract_version())
#             print("text:",text)
#             # clean tesseract text by removing any unwanted blank spaces
#             # clean_text = re.sub('[\W_]+', '', text)
#             # print("clean_text:",clean_text,"lenc:",len(clean_text))
#             if len(text)<=4:
#                 plate_num += "".join(text.split())
#         except:
#             text = None
#         # if text != None and  len(text)==1:
#         #     plate_num+=text
#     if plate_num != None :
#         print("License Plate #: ", plate_num)
#     #cv2.imshow("Character's Segmented", im2)
#     #cv2.waitKey(0)
#     return plate_num
#
# def character_recongition_easyocr(plate):
#     gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
#     gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
#     gray = cv2.GaussianBlur(gray, (5,5), 0)
#     gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#     kernel = np.ones((3, 3), np.uint8)
#     gray = cv2.bitwise_not(gray)
#     gray = cv2.erode(gray, kernel)
#     gray = cv2.bitwise_not(gray)
#
#     # # plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
#     # # plt.show()
#     # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
#     # print('thresh:',thresh.shape)
#     # rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
#     # dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
#     # try:
#     #     contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     # except:
#     #     ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     # sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
#     # cv2.drawContours(gray, sorted_contours, -1, (0,255,0), 1)
#
#     region_threshold=0.1
#     reader = easyocr.Reader(['fa'])
#     ocr_result = reader.readtext(gray)
#     plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
#     plt.show()
#     text = filter_text(gray, ocr_result, region_threshold)
#     print('text:',text)
#     return text
#
#
# def filter_text(region, ocr_result, region_threshold):
#     rectangle_size = region.shape[0]*region.shape[1]
#
#     plate = []
#     for result in ocr_result:
#         length = np.sum(np.subtract(result[0][1], result[0][0]))
#         height = np.sum(np.subtract(result[0][2], result[0][1]))
#
#         if length*height / rectangle_size > region_threshold:
#             plate.append(result[1])
#     return plate
