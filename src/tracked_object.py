import uuid
import time
import numpy as np
# from PIL import Image
# from constant import OPENCV_OBJECT_TRACKERS

FMT_TRACKER = 'tracker'
"""
Origin top left.
Coordinates for box is (x, y, w, h) where
    x_min = x
    x_max = x + w
    y_max = image-height - y
    y_min = y_max - h
    see (see https://docs.opencv.org/3.4/d2/d44/classcv_1_1Rect__.html)
"""
FMT_TF_BBOX = 'tensorflow_boundingbox'
"""
Origin is top left
Coordinates for box is (y_min, x_min, y_max, x_max)
"""
FMT_STANDARD = 'std'
"""
Origin is top left
Coordinates for box is (x_min, y_min, x_max, y_max)
"""


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
            image_height : required
            image_width : required
            fmt: string, one of FMT_TRACKER, FMT_TF_BBOX, FMT_STANDARD ;
            use_normalized_coordinates : True means coordinates are 0<x<1
                False is absolute pixels. True when scoring from TF model
        """
        self.image_height = image_height
        self.image_width = image_width
        if fmt == FMT_TRACKER:
            (x, y, w, h) = box
            self.x_min = x
            self.x_max = x + w
            self.y_min = y
            self.y_max = y + h
        elif fmt == FMT_TF_BBOX:
            (self.y_min, self.x_min, self.y_max, self.x_max) = box
        elif fmt == FMT_STANDARD:
            (self.x_min, self.y_min, self.x_max, self.y_max) = box
        if use_normalized_coordinates:
            if image_height is None or image_width is None:
                raise Exception('Need to specify image width and height in constuctor')
            else:
                self.x_min = int(self.x_min * image_width)
                self.x_max = int(self.x_max * image_width)
                self.y_min = int(self.y_min * image_height)
                self.y_max = int(self.y_max * image_height)
        self.validate_asserts()

    def get_bbox(self,
                 fmt='tracker',
                 use_normalized_coordinates=False):
        """
        Description:
            Returns a tuple of the values in the cv2.trackers format
        Args:
            fmt: string, one of FMT_TRACKER, FMT_TF_BBOX, FMT_STANDARD
            use_normalized_coordinates : bool, for tf_box and standard format only
        Returns
            format = tuple (a,b,c,d) with coordinates in the proper format
        """
        self.validate_asserts()
        if fmt == FMT_TRACKER:
            if use_normalized_coordinates:
                x = self.x_min / self.image_width
                y = self.y_min / self.image_height
                w = (self.x_max - self.x_min) / self.image_width
                h = (self.y_max - self.y_min) / self.image_height
            else:
                x = self.x_min
                y = self.y_min
                w = self.x_max - self.x_min
                h = self.y_max - self.y_min
            assert x >= 0, "problem - X < 0"
            assert y >= 0, "problem - Y < 0"
            assert w >= 0, "problem - w < 0"
            assert h >= 0, "problem - h < 0"
            return (x, y, w, h)
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
                        self.y_min / self.image_height,
                        self.x_max / self.image_width,
                        self.y_max / self.image_height)
            else:
                return (self.x_min, self.y_min, self.x_max, self.y_max)

    def update(self,
               box: [int],
               fmt='tracker'):
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
            self.y_min = y
            self.y_max = y + h
        elif fmt == FMT_TF_BBOX:
            (self.y_min, self.x_min, self.y_max, self.x_max) = box
        elif fmt == FMT_STANDARD:
            (self.x_min, self.y_min, self.x_max, self.y_max) = box
        self.validate_asserts()

    def validate_asserts(self):
        """
        Description : used to validate coordinates are ok
            at creation and upon update
        Args: None
        Returns : None
        """
        assert self.x_min >= 0, "problem - x_min < 0"
        assert self.x_max >= 0, "problem - x_max < 0"
        assert self.x_min <= self.x_max, "problem, x_max < x_min"
        assert self.y_min >= 0, "problem - y_min < 0"
        assert self.y_max >= 0, "problem - y_max < 0"
        assert self.y_min <= self.y_max, "problem y_max < y_min"


class TrackedObjectMP(object):
    """
    Used to track objects that are seen by the camera -
    passed to the multiprocessing task without the opencv2 object
    because it can't be pickled.
    """
    def __init__(self,
                 object_class: int,
                 score: float,
                 original_image_resolution: tuple,
                 box: tuple,
                 distance=-1,
                 coords_3d_coordinates_center_point=None,
                 fmt='std',
                 use_normalized_coordinates=False):
        """
        Args
            object_class = int representing the object class being tracked
            score = float for prediction score for the object (0=<x<=1)
            original_image_resolution: tuple (height, width) of unscaled image
                because image input may be already resized
            box = (a,b,c,d) depending on fmt for new object
            distance = float, distance in millimeter from kinect camera
            coords_3d_coordinates_center_point : tuple (x,y,z) with respect to
                the camera coordinate system (see self.k4a_config). This gets
                updated by the system.
            fmt: string, one of FMT_TRACKER, FMT_BBOX, FMT_STANDARD
            use_normalized_coordinates: bool, normalized coordinates are
                relative to image, that would be in the tracker bounding box
        """
        self.id = uuid.uuid4()
        self.object_class = object_class
        self.last_seen = time.time()
        self.score = score
        self.distance = distance
        self.coords_3d_coordinates_center_point = None
        self._original_image_resolution = original_image_resolution
        height, width = self._original_image_resolution
        self._bounding_box = BoundingBox(
            box=box,
            image_height=height,
            image_width=width,
            fmt=fmt,
            use_normalized_coordinates=use_normalized_coordinates)

    def get_max_overlap_bb(self,
                           list_bb: [BoundingBox]):
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

    def get_overlap_bb(self,
                       bb: BoundingBox) -> float:
        """
        Description:
            Gets the overlap between two bounding boxes (self and bb)
        Args:
            bb:Bounding box to be compared with self.tracker_bounding_box
        Returns:
            Area that overlaps (in pixels^2)
        """
        overlap_x = max(
            0, min(self._bounding_box.x_max, bb.x_max) - max(
                self._bounding_box.x_min, bb.x_min))
        overlap_y = max(
            0, min(self._bounding_box.y_max, bb.y_max) - max(
                self._bounding_box.y_min, bb.y_min))
        overlap_area = max(0, float(overlap_x) * float(overlap_y))
        return overlap_area

    def set_bbox(self,
                 bbox: BoundingBox):
        # fmt='tracker'):
        """
        Description : exposes bounding box member 'update' to track
            update its coordinates while tracking last_seen time
        Args:
            bbox: BoundingBox representing the new bounding box
            # fmt: string, one of FMT_TRACKER, FMT_BBOX, FMT_STANDARD
        """
        fmt = FMT_TRACKER
        self.last_seen = time.time()
        self._bounding_box.update(
            bbox.get_bbox(
                fmt=fmt,
                use_normalized_coordinates=False),
            fmt=fmt)

    def get_bbox(self,
                 fmt='tracker',
                 use_normalized_coordinates=False):
        """
        Description :
            exposes bounding box member to retrieve the bounding box
        Args:
            fmt: string, one of FMT_TRACKER, FMT_BBOX, FMT_STANDARD
        """
        return self._bounding_box.get_bbox(
            fmt=fmt,
            use_normalized_coordinates=use_normalized_coordinates)
