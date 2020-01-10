import numpy as np
import os
import sys
import argparse
import tarfile
import tensorflow as tf
import zipfile
import cv2

# from collections import defaultdict
# from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util

# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(description="Inference Program")
    parser.add_argument("--camera_index", help="0 = front cam for surface book, 1 = back cam for surface book, 2 = usb cam 1, 3 = usb cam 2, etc.", type=int)
    parser.add_argument("--frozen_graph_path", help="path to frozen detection graph", type=str)
    parser.add_argument("--label_map_path", type=str, help='path to label_map.pbtxt')
    parser.add_argument('--num_labels', type=int, help='number of labels - it is written in the label_map.pbtxt but I have been lazy')

    args = parser.parse_args()

    # Define the video stream
    cap = cv2.VideoCapture(args.camera_index)
    PATH_TO_CKPT = args.frozen_graph_path
    PATH_TO_LABELS = args.label_map_path
    NUM_CLASSES = args.num_labels

    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, 
    # we know that this corresponds to `airplane`.  Here we use internal utility functions, 
    # but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Detection
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while True:
                # Read frame from camera
                ret, image_np = cap.read()
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Extract image tensor
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Extract detection boxes
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Extract detection scores
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                # Extract detection classes
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                # Extract number of detectionsd
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                # Display output
                cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

if __name__ == '__main__':
    main()