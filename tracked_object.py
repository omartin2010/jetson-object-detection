import uuid
import time
import numpy as np
# import cv2


class BoundingBox(object):
    """
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
        Constructor for bounding box
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


class TrackedObject(object):
    """
    Used to track objects that are seen by the camera
    """
    def __init__(self,
                 tracker_bbox,
                 object_class: str,
                 score: float):
        """
        Args
            tracker_bbox = (x, y, w, h) value for current
                bounding box for this object
            object_class = string representing the object class being tracked
            score = prediction score for the object
        """
        self._object_id = uuid.uuid4()
        self._object_class = object_class
        self._last_seen = time.time()
        self._score = score
        self._tracker_bounding_box = BoundingBox(
            tracker_bbox[0],
            tracker_bbox[1],
            tracker_bbox[0] + tracker_bbox[2],
            tracker_bbox[1] + tracker_bbox[3],
            use_normalized_coordinates=False)
        # add box position and filter between bb and tracker position

    @property
    def last_seen(self):
        return self._last_seen

    @last_seen.setter
    def last_seen(self, value: time.time()):
        self._last_seen = value

    def get_max_overlap_bb(self, list_bb):
        """
        Get the bounding box that has the largets overlap
        """
        areas = []
        for bb in list_bb:
            area = self.overlap_bb(bb)
            areas.append(area)
        if np.sum(areas) > 0:
            result = np.argmax(np.array(areas))
        else:
            result = None
        return result

    def overlap_bb(self, bb) -> float:
        overlap_x = max(
            0, min(self._tracker_bounding_box.x_max, bb.x_max) - max(
                self._tracker_bounding_box.x_min, bb.x_min))
        overlap_y = max(
            0, min(self._tracker_bounding_box.y_max, bb.y_max) - max(
                self._tracker_bounding_box.y_min, bb.y_min))
        overlap_area = max(0, float(overlap_x) * float(overlap_y))
        return overlap_area
