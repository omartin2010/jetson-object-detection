# flake8: ignore=E501
from logger import RoboLogger
from constant import LOGGER_OBJECT_DETECTOR_STARTUP
from detector import ObjectDetector
import os
import time
import sys
import argparse
import logging
import json
import traceback
import signal


log = RoboLogger.getLogger()
log.warning(LOGGER_OBJECT_DETECTOR_STARTUP,
            msg="Initial imports are completed.")
objectDetector = None


def main():
    """
    Full Jetson Detector Program
    """

    parser = argparse.ArgumentParser(description="Inference Program")
    parser.add_argument(
        "--config_file", help="json configuration file containint params to initialize the jetson", type=str)
    args = parser.parse_args()

    try:
        global objectDetector

        if not os.path.exists(args.config_file):
            raise ValueError(
                f'Cannot find configuration file "{args.config_file}"')

        with open(args.config_file, 'r') as f:
            robotJetsonConfiguration = json.load(f)

        log.warning(LOGGER_OBJECT_DETECTOR_STARTUP,
                    msg='Launching object detector now.')
        signal.signal(signal.SIGTERM, sigterm_handler)
        objectDetector = ObjectDetector(robotJetsonConfiguration)
        log.warning(LOGGER_OBJECT_DETECTOR_STARTUP,
                    msg='Launching runner.')
        objectDetector.run()

    except SystemExit:
        log.info(LOGGER_OBJECT_DETECTOR_STARTUP, 'Caught SystemExit...')
    except Exception:
        log.critical(LOGGER_OBJECT_DETECTOR_STARTUP,
                     'Crash in startup : {}'.format(traceback.print_exc()))
    finally:
        graceful_shutdown()
        logging.shutdown()


def sigterm_handler(sig, frame):
    log.info(LOGGER_OBJECT_DETECTOR_STARTUP,
             'SIGTERM caught. Docker Container being terminated.')
    graceful_shutdown()
    log.info(LOGGER_OBJECT_DETECTOR_STARTUP, 'SIGTERM signal processing done.')


def graceful_shutdown():
    try:
        log.info(LOGGER_OBJECT_DETECTOR_STARTUP,
                 'Initiating Graceful Shutdown')
        global bot
        if 'bot' in globals():
            log.info(LOGGER_OBJECT_DETECTOR_STARTUP, 'Deleting robot... now.')
            bot.__del__()
            del bot
        sys.exit(0)
    except SystemExit:
        log.info(LOGGER_OBJECT_DETECTOR_STARTUP, 'Exiting process.')
    except Exception:
        log.critical(LOGGER_OBJECT_DETECTOR_STARTUP,
                     'trace: {}'.format(traceback.print_exc()))


if __name__ == "__main__":
    main()
    time.sleep(10)
    log.info(LOGGER_OBJECT_DETECTOR_STARTUP, "Done")
    try:
        sys.exit(0)
    except SystemExit:
        log.info(LOGGER_OBJECT_DETECTOR_STARTUP, 'Exiting startup.')
