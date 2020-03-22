import uuid
import time
import numpy as np
from constant import OPENCV_OBJECT_TRACKERS


class BoundingBox(object):
    """
    Description:
        Used to track bounding boxes and openCV tracker coordinates
    """
    def __init__(self,
                 x_min: int,
                 y_min: int,
                 x_max: int,
                 y_max: int,
                 use_normalized_coordinates=True,
                 image_height=None,
                 image_width=None):
        """
        Description:
            Constructor for bounding box
        Args:
            x_min : x_min coordinate
            y_min : y_min coordinate
            x_max : x_max coordinate
            y_max : y_max coordinate
            use_normalized_coordinates : True means coordinates are 0<x<1
                False is absolute pixels
            image_height : if normalized_coordinates = False, this is required
            image_width : if normalized_coordinates = False, this is required
        """
        if use_normalized_coordinates:
            if image_height is None or image_width is None:
                raise('Need to specify image width and height in constuctor')
            else:
                self.x_min = int(x_min * image_width)
                self.x_max = int(x_max * image_width)
                self.y_min = int(y_min * image_height)
                self.y_max = int(y_max * image_height)
        else:
            self.x_min = x_min
            self.x_max = x_max
            self.y_min = y_min
            self.y_max = y_max

    def get_cv2_tracker_fmt(self):
        """
        Description:
            Returns a tuple of the values in the cv2.trackers format
        Args:
            None
        Returns
            format = (x,y,w,h) tuple with coordinates in the proper format
        """
        return (self.x_min,
                self.y_min,
                self.x_max - self.x_min,
                self.y_max - self.y_min)


class TrackedObject(object):
    """
    Used to track objects that are seen by the camera
    """
    def __init__(self,
                 object_class: str,
                 score: float,
                 image,
                 tracker_bbox: list,
                 tracker_alg='kcf',
                 use_normalized_coordinates=False):
        """
        Args
            object_class = string representing the object class being tracked
            score = float for prediction score for the object (0=<x<=1)
            image = numpy array containing image ti initialize the tracker
            tracker_bbox = (x_min, y_min, x_max, y_max) value for current
                bounding box for this object
            tracker_alg : string, one of the values in OPENCV_OBJECT_TRACKERS
            use_normalized_coordinates: bool, normalized coordinates are
                relative to image, that would be in the tracker bounding box
        """
        self._object_id = uuid.uuid4()
        self.object_class = object_class
        self.last_seen = time.time()
        self.score = score
        self.tracker = OPENCV_OBJECT_TRACKERS[tracker_alg]()
        self.tracker_bounding_box = BoundingBox(
            x_min=tracker_bbox[0], y_min=tracker_bbox[1],
            x_max=tracker_bbox[2], y_max=tracker_bbox[3],
            use_normalized_coordinates=use_normalized_coordinates)
        self.tracker.init(
            image, self.tracker_bounding_box.get_cv2_tracker_fmt())

    def get_max_overlap_bb(self, list_bb: [BoundingBox]):
        """
        Description:
            Gets the bounding box that has the largets overlap
        Args:
            list_bb: list of BoundingBox to compare
        Returns:
            value for the index of the item in the list that overlaps most
        """
        areas = []
        for bb in list_bb:
            area = self.get_overlap_bb(bb)
            areas.append(area)
        if np.sum(areas) > 0:
            result = np.argmax(np.array(areas))
        else:
            result = None
        return result

    def get_overlap_bb(self, bb: BoundingBox) -> float:
        """
        Description:
            Gets the overlap between two bounding boxes (self and bb)
        Args:
            bb:Bounding box to be compared with self.tracker_bounding_box
        Returns:
            Area that overlaps (in pixels^2)
        """
        overlap_x = max(
            0, min(self.tracker_bounding_box.x_max, bb.x_max) - max(
                self.tracker_bounding_box.x_min, bb.x_min))
        overlap_y = max(
            0, min(self.tracker_bounding_box.y_max, bb.y_max) - max(
                self.tracker_bounding_box.y_min, bb.y_min))
        overlap_area = max(0, float(overlap_x) * float(overlap_y))
        return overlap_area

    def update(self, image):
        """
        Description:
            Updates all trackers that are being followed
        Args:
            image: numpy array of image to update trackers with
        """
        (box, success) = self.tracker.update(image)
