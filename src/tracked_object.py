import uuid
import time
import numpy as np
from PIL import Image
from constant import OPENCV_OBJECT_TRACKERS

FMT_TRACKER = 'tracker'
FMT_TF_BBOX = 'tensorflow_boundingbox'
FMT_STANDARD = 'std'


class BoundingBox(object):
    """
    Description:
        Used to track bounding boxes and openCV tracker coordinates
    """

    def __init__(self,
                 box: tuple,
                 image_height: int,
                 image_width: int,
                 fmt='tensorflow_boundingbox',
                 use_normalized_coordinates=True):
        """
        Description:
            Constructor for bounding box with origin bottom left.
        Args:
            box : tuple of proper format depending on fmt below
            fmt: string, one of FMT_TRACKER, FMT_TF_BBOX, FMT_STANDARD ;
                FMT_TRACKER has origin top left
                FMT_TF_BBOX and FMT_STANDARD has origin bottom left
            image_height : required
            image_width : required
            use_normalized_coordinates : True means coordinates are 0<x<1
                False is absolute pixels
        """
        self.image_height = image_height
        self.image_width = image_width
        if fmt == FMT_TRACKER:
            x, y, w, h = box
            self.x_min = x
            self.x_max = x + w
            self.y_max = self.height - y
            self.y_min = self.y_max - self.height
        elif fmt == FMT_TF_BBOX:
            self.y_min, self.x_min, self.y_max, self.x_max = box
        elif fmt == FMT_STANDARD:
            self.x_min, self.x_max, self.y_min, self.y_max = box
        if use_normalized_coordinates:
            if image_height is None or image_width is None:
                raise('Need to specify image width and height in constuctor')
            else:
                self.x_min = int(self.x_min * image_width)
                self.x_max = int(self.x_max * image_width)
                self.y_min = int(self.y_min * image_height)
                self.y_max = int(self.y_max * image_height)

    def get_bbox(self, fmt='tracker', use_normalized_coordinates=False):
        """
        Description:
            Returns a tuple of the values in the cv2.trackers format
        Args:
            fmt: string, one of FMT_TRACKER, FMT_TF_BBOX, FMT_STANDARD ;
                FMT_TRACKER is x_min, image_height - y_max, width, height
                (in our coordinate syetem). FMT_TRACKER has origin top left.
                (see https://docs.opencv.org/3.4/d2/d44/classcv_1_1Rect__.html)
                FMT_STD has origin bottom left.
                FMT_TF_BBOX has origin bottom left - but order is ymin, xmin, ymax, ymin
            use_normalized_coordinates : bool, for tf_box and standard format only
        Returns
            format = tuple (a,b,c,d) with coordinates in the proper format
        """
        if fmt == FMT_TRACKER:
            return (self.x_min, self.image_height - self.y_max,
                    self.x_max - self.x_min,
                    self.y_max - self.y_min)
        elif fmt == FMT_TF_BBOX:
            if use_normalized_coordinates:
                return (self.y_min / self.image_height,
                        self.x_min / self.image_width,
                        self.y_max / self.image_height,
                        self.x_max / self.image_width)
            else:
                return (self.y_min, self.x_min, self.y_max, self.x_max)
        elif fmt == FMT_STANDARD:
            if use_normalized_coordinates:
                return (self.x_min / self.image_width,
                        self.x_max / self.image_width,
                        self.y_min / self.image_height,
                        self.y_max / self.image_height)
            else:
                return (self.x_min, self.x_max, self.y_min, self.y_max)

    def update(self, box: [int], fmt='tracker'):
        """
        Description:
            Update position with various formatting (see input param fmt)
        Args:
            bbox : list if ints in the x, y, w, h format
            fmt: string, one of FMT_TRACKER, FMT_BBOX, FMT_STANDARD
        """
        if fmt == FMT_TRACKER:
            """
            FMT_TRACKER = x, y : top left corner, w, h = width + height
                opencv coordinate system has origin top left
            """
            (x, y, w, h) = box
            self.x_min = x
            self.x_max = x + w
            self.y_max = self.image_height - y
            self.y_min = self.y_max - h
        elif fmt == FMT_TF_BBOX:
            (self.y_min, self.x_min, self.y_max, self.x_max) = box
        elif fmt == FMT_STANDARD:
            (self.x_min, self.y_min, self.x_max, self.y_max) = box


class TrackedObject(object):
    """
    Used to track objects that are seen by the camera
    """
    def __init__(self,
                 object_class: int,
                 score: float,
                 image,
                 tracker_bbox: tuple,
                 fmt='std',
                 resized_image_resolution=(300, 300),
                 tracker_alg='kcf',
                 use_normalized_coordinates=False):
        """
        Args
            object_class = int representing the object class being tracked
            score = float for prediction score for the object (0=<x<=1)
            image = numpy array containing image ti initialize the tracker
            tracker_bbox = (x_min, y_min, x_max, y_max) value for current
                bounding box for this object (FMT_STANDARD)
            resized_image_resolution = (x, y) resolution (300x300) by default
                to accelerate running the tracker
            tracker_alg : string, one of the values in OPENCV_OBJECT_TRACKERS
            use_normalized_coordinates: bool, normalized coordinates are
                relative to image, that would be in the tracker bounding box
        """
        self.id = uuid.uuid4()
        self.object_class = object_class
        self.last_seen = time.time()
        self.score = score
        self._resized_image_resolution = resized_image_resolution
        self.tracker = OPENCV_OBJECT_TRACKERS[tracker_alg]()
        height, width = image.shape[:2]
        self.bounding_box = BoundingBox(
            box=tracker_bbox,
            image_height=height,
            image_width=width,
            fmt=fmt,
            use_normalized_coordinates=use_normalized_coordinates)
        if image.shape[:2] is not self._resized_image_resolution:
            image = np.asarray(Image.fromarray(image).resize(self._resized_image_resolution))
        self.tracker.init(
            image, self.bounding_box.get_bbox())

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
            0, min(self.bounding_box.x_max, bb.x_max) - max(
                self.bounding_box.x_min, bb.x_min))
        overlap_y = max(
            0, min(self.bounding_box.y_max, bb.y_max) - max(
                self.bounding_box.y_min, bb.y_min))
        overlap_area = max(0, float(overlap_x) * float(overlap_y))
        return overlap_area

    def update(self, image=None, box=None, fmt=None):
        """
        Description:
            Updates all trackers that are being followed
        Args:
            image: numpy array of image to update trackers with
            box: (a,b,c,d) tuple coordinates of a bounding box with fmt
            fmt: string, one of FMT_TRACKER, FMT_BBOX, FMT_STANDARD
        """
        self.last_seen = time.time()
        if image is not None:
            """
            update with cv2 tracker functions
            """
            if image.shape[:2] is not self._resized_image_resolution:
                image = np.asarray(Image.fromarray(image).resize(self._resized_image_resolution))
            (success, box) = self.tracker.update(image)
            if success:
                self.bounding_box.update(box)
        elif box is not None:
            """
            updated box as per new detection
            """
            self.bounding_box.update(box, fmt=fmt)
        else:
            raise('Image is not defined and bbox is not defined.')
