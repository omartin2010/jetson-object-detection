import os
import multiprocessing as mp
from functools import wraps
from logger import RoboLogger
from constant import LOGGER_OBJECT_DETECTION_PUBLISH_QUEUES_UNREGISTER, \
    LOGGER_OBJECT_DETECTION_PUBLISH_QUEUES_REGISTER
log = RoboLogger.getLogger()


def ensure_parent(func):
    @wraps(func)
    def inner(self, *args, **kwargs):
        if os.getpid() != self._creator_pid:
            raise RuntimeError("{} can only be called in the "
                               "parent.".format(func.__name__))
        return func(self, *args, **kwargs)
    return inner


class PublishQueue(object):
    def __init__(self):
        self._queues = {}
        self._creator_pid = os.getpid()
        self.lock = mp.Lock()

    def __getstate__(self):
        self_dict = self.__dict__
        self_dict['_queues'] = {}    # []
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    @ensure_parent
    def register(self, name):
        with self.lock:
            q = mp.Manager().Queue()
            log.info(LOGGER_OBJECT_DETECTION_PUBLISH_QUEUES_REGISTER,
                     msg=f'Registering {name[:8]} to queues')
            self._queues[name] = q
            return q

    @ensure_parent
    def unregister(self, name):
        with self.lock:
            log.info(LOGGER_OBJECT_DETECTION_PUBLISH_QUEUES_UNREGISTER,
                     msg=f'Unregistering {name[:8]} from queues')
            self._queues.pop(name)

    @ensure_parent
    def publish(self, val):
        with self.lock:
            for (uuid, q) in self._queues.items():
                q.put(val)
