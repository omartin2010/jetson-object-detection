from multiprocessing import Process, Event
import multiprocessing as mp
import time
import os
import queue
from logger import RoboLogger
import numpy as np
import traceback
from publish_queues import PublishQueue
from tracked_object import TrackedObjectMP, BoundingBox, FMT_TRACKER
from constant import LOGGER_OBJECT_DETECTION_PROCESS_TRACK_OPENCV_OBJECT, \
    OPENCV_OBJECT_TRACKERS
log = RoboLogger.getLogger()


class ObjectTrackingProcess(Process):
    """
    Description:
        Class runs a process in per tracked object. This is required to offload
            main CPU from the open CV work.
    """
    def __init__(self,
                 tracked_object: TrackedObjectMP,
                 tracker_alg: str,
                 initial_image: np.array,
                 frame_duration: float,
                 image_queue: PublishQueue,
                 new_tf_info_queue: mp.Queue,
                 tracking_object_queue: mp.Queue):
        """
        Description:
            Constructor for the class.
        Args:
            tracked_object: <class 'TrackedObjectMP()'>, object tracked by process
            tracker_alg : string, one of the values in OPENCV_OBJECT_TRACKERS
            initial_image : numpy array containing the resized image that is used to
                initialize the tracker
            frame_duration : float, corresponds to the 1/FPS of input camera
            image_queue: <class 'multiprocessing.Queue()'> : contains images
                that are pushed on it in numpy format when a new image was read
                from the video capture
            tracking_object_queue: <class 'multiprocessing.Queue()'> : queue to
                publish the updated tracked_object_mp (incl. new bbox)
            new_tf_info_queue : <class 'multiprocessing.Queue()'> : queue to contain
                new tf_box, scores and classes discovered in the scoring thread to
                refresh current bounding box that might have drifted over time.
        """
        Process.__init__(self)
        self.tracked_object = tracked_object
        self.tracker_alg = tracker_alg
        self.initial_image = initial_image
        self.frame_duration = frame_duration
        self.image_queue = image_queue
        self.tracking_object_queue = tracking_object_queue
        self.new_tf_info_queue = new_tf_info_queue
        self.exit = Event()

    def run(self):
        """
        Description:
            Main function that runs the process.
        """
        try:
            log.warning(LOGGER_OBJECT_DETECTION_PROCESS_TRACK_OPENCV_OBJECT,
                        f'Launching process {os.getpid()} for object ID '
                        f'{self.tracked_object.id}')
            exitcode = 0
            tracker = OPENCV_OBJECT_TRACKERS[self.tracker_alg]()
            _ = tracker.init(self.initial_image, self.tracked_object.get_bbox())
            logging_loops = 100
            min_time_bw_identical_warnings = 1
            n_loops = 0
            n_queue_errors = 0
            new_bbox = None
            last_warning_message_time = time.time()
            while not self.exit.is_set():
                start_time = time.time()
                try:
                    image = self.image_queue.get(block=True, timeout=2)
                except queue.Empty:
                    n_queue_errors += 1
                    log.critical(LOGGER_OBJECT_DETECTION_PROCESS_TRACK_OPENCV_OBJECT,
                                 msg=f'Empty image queue. Strange... Investigate')
                    if n_queue_errors >= 5:
                        log.critical(LOGGER_OBJECT_DETECTION_PROCESS_TRACK_OPENCV_OBJECT,
                                     msg=f'Empty image queue for more than 5 loops : '
                                         f'Exiting process.')
                        self.exit.set()
                height, width = image.shape[:2]
                # Get new coordinates of a scored image (via TF model scoring)
                try:
                    new_bbox = None
                    new_score = None
                    new_class = None
                    new_bbox, new_score, new_class = self.new_tf_info_queue.get_nowait()
                    tracker = None
                    tracker = OPENCV_OBJECT_TRACKERS[self.tracker_alg]()
                    tracker.init(image, new_bbox.get_bbox())
                    self.tracked_object.set_bbox(new_bbox)
                    self.tracked_object.score = new_score
                    self.tracked_object.object_class = new_class
                    log.warning(LOGGER_OBJECT_DETECTION_PROCESS_TRACK_OPENCV_OBJECT,
                                msg=f'Getting new position from tensorflow for '
                                    f'object {self.tracked_object.id}')
                # If no TF scoring, queue is Empty ==> use tracker to find position
                except queue.Empty:
                    # Costly operation - update opencv tracker information
                    (success, bbox) = tracker.update(image)
                    if success:
                        self.tracked_object.set_bbox(
                            BoundingBox(
                                box=bbox,
                                image_height=height,
                                image_width=width,
                                fmt=FMT_TRACKER,
                                use_normalized_coordinates=False))
                        # fmt=FMT_TRACKER)
                    elif new_bbox:
                        now = time.time()
                        if now - last_warning_message_time > min_time_bw_identical_warnings:
                            last_warning_message_time = now
                            log.warning(
                                LOGGER_OBJECT_DETECTION_PROCESS_TRACK_OPENCV_OBJECT,
                                msg=f'Trying to reinitialize tracker for object '
                                    f'{self.tracked_object.id} with last bounding box info.')
                        tracker = None
                        tracker = OPENCV_OBJECT_TRACKERS[self.tracker_alg]()
                        tracker.init(image, new_bbox.get_bbox())
                except Exception:
                    raise Exception(f'Problem : {traceback.print_exc()}')
                # Dumping tracked_object to queue - unless exit flag unset
                if not self.exit.is_set():
                    # put box, score and class not to overried other metadata held in other process
                    self.tracking_object_queue.put(self.tracked_object, block=True, timeout=2)
                # Sleep if required
                loop_duration = time.time() - start_time
                time.sleep(max(0, self.frame_duration - loop_duration))
                if n_loops % logging_loops == 0:
                    log.debug(
                        LOGGER_OBJECT_DETECTION_PROCESS_TRACK_OPENCV_OBJECT,
                        msg=f'Process still tracking {self.tracked_object.id} '
                            f'after {n_loops} loops - object class = '
                            f'{self.tracked_object.object_class}.')
                n_loops += 1
        except queue.Full:
            log.critical(LOGGER_OBJECT_DETECTION_PROCESS_TRACK_OPENCV_OBJECT,
                         msg=f'Output queue full (object {self.tracked_object.id}).'
                             f'See thread_poll_object_tracking_process_queue '
                             f'to see what might be wrong.')
            exitcode = 2
        except:
            raise Exception(f'Problem in process_track_opencv_object: '
                            f'{traceback.print_exc()}')
            exitcode = 3
        finally:
            log.warning(LOGGER_OBJECT_DETECTION_PROCESS_TRACK_OPENCV_OBJECT,
                        msg=f'Process {os.getpid()} done for object {self.tracked_object.id}')
        return exitcode

    def shutdown(self):
        """
        Description
            Call to shut down the main process. Will send an event to it.
        """
        log.warning(
            LOGGER_OBJECT_DETECTION_PROCESS_TRACK_OPENCV_OBJECT,
            msg=f'Kill signal received for process handling '
                f'object {self.tracked_object.id}')
        self.exit.set()
