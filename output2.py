# -*- coding: utf-8 -*-

#Imports
import time
import numpy as np
import os
import tensorflow as tf
import cv2
from PIL import Image
import pandas as pd

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


#Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def tf_dete(pth=os.path.dirname(os.path.abspath(__file__))):
    start = time.time()
    #path=os.path.normpath(pth+'../../../vision_output/output')
    path=pth
    #print(path)
    os.chdir(path)
  
    #print(os.path.normpath(path+'/../graph4'))



    MODEL_NAME =  os.path.normpath(path+'/../graph16')


    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('./my_label_map.pbtxt')
    
    NUM_CLASSES = 48
    
    
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')    
            

    #Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    
    #Detection
    # For the sake of simplicity we will use only 2 images:
    # image1.jpg
    # image2.jpg
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    PATH_TO_TEST_IMAGES_DIR = (path+'/test_image')
    os.chdir(PATH_TO_TEST_IMAGES_DIR)
    TEST_IMAGE_PATHS = os.listdir(PATH_TO_TEST_IMAGES_DIR)
    
    
    output_image_path = (path+'/result/')
    output_csv_path = (path+'/result/')

    #for image_folder in TEST_IMAGE_DIRS:
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            #TEST_IMAGE_PATHS = os.listdir(os.path.join(image_folder))
            #os.makedirs(output_image_path+image_folder)
            #data = pd.DataFrame()
            for image_path in TEST_IMAGE_PATHS:
                data = pd.DataFrame()
                image = Image.open(image_path)
                width, height = image.size
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                s_score=0.8
                s_boxes = boxes[scores > s_score]
                s_classes = classes[scores > s_score]
                s_scores=scores[scores>s_score]
                #print(s_boxes)
                vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        min_score_thresh=s_score,
                        line_thickness=4)
                #write images
                
                
                cv2.imwrite(output_image_path+image_path.split('\\')[-1],cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB))
                #print(output_image_path+image_path.split('\\')[-1]) 
                '''
                s_boxes = boxes[scores > 0.95]
                s_classes = classes[scores > 0.95]
                s_scores=scores[scores>0.95]
                '''
                #write table
                
                for i in range(len(s_classes)):
                    
                    newdata= pd.DataFrame(0, index=range(1), columns=range(7))
                    newdata.iloc[0,0] = image_path.split("\\")[-1].split('.')[0]
                    newdata.iloc[0,1] = s_boxes[i][0]*height  #ymin
                    newdata.iloc[0,2] = s_boxes[i][1]*width     #xmin
                    newdata.iloc[0,3] = s_boxes[i][2]*height    #ymax
                    newdata.iloc[0,4] = s_boxes[i][3]*width     #xmax
                    newdata.iloc[0,5] = s_scores[i]
                    newdata.iloc[0,6] = s_classes[i]
                    
                    data = data.append(newdata)
                data.to_csv(output_csv_path+image_path+'.csv',index = False)
                #print(output_csv_path+image_path+'.csv')
                data.empty 
                
    end =  time.time()
    print("Execution Time: ", end - start)
    
