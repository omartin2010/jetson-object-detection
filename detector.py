from logger import RoboLogger
import numpy as np
import tensorflow as tf
import cv2
from utils import label_map_util
from utils import visualization_utils as vis_util
from constant import LOGGER_OBJECT_DETECTOR_MAIN, LOGGER_OBJECT_DETECTOR_LOAD_MODEL

log = RoboLogger.getLogger()
log.warning(LOGGER_OBJECT_DETECTOR_MAIN, msg="Initial imports are completed.")


class ObjectDetector(object):
    """ main class for the object detector running on the jetson """

    def __init__(self, camera_index: int, frozen_graph_path: str, label_map_path: str, num_classes: int) -> None:
        self._camera_index = camera_index
        self.frozen_graph_path = frozen_graph_path
        self.num_classes = num_classes
        self.label_map_path = label_map_path

    def load_model(self) -> None:
        log.info(LOGGER_OBJECT_DETECTOR_LOAD_MODEL, f'Connecting to camera {self._camera_index}')
        self.cap = cv2.VideoCapture(self._camera_index)

        log.info(LOGGER_OBJECT_DETECTOR_LOAD_MODEL, f'Rehydrating inference graph in memory...')

        # Load a (frozen) Tensorflow model into memory.
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.frozen_graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Loading label map
        # Label maps map indices to category names, so that when our convolution network predicts `5`,
        # we know that this corresponds to `airplane`.  Here we use internal utility functions,
        # but anything that returns a dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(self.label_map_path)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=self.num_classes,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def run_detection_loop(self, show_video=False) -> None:

        # Detection
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                while True:
                    # Read frame from camera
                    ret, image_np = self.cap.read()
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    # Extract image tensor
                    image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                    # Extract detection boxes
                    boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Extract detection scores
                    scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                    # Extract detection classes
                    classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                    # Extract number of detections
                    num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    if show_video:
                        # Visualization of the results of a detection.
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            image_np,
                            np.squeeze(boxes),
                            np.squeeze(classes).astype(np.int32),
                            np.squeeze(scores),
                            self.category_index,
                            use_normalized_coordinates=True,
                            line_thickness=8)
                        # Display output
                        cv2.imshow('object detection', cv2.resize(image_np, (1024, 576)))
                        if cv2.waitKey(5) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            break

        def __del__(self) -> None:
            pass
