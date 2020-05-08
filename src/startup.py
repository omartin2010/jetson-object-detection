# flake8: ignore=E501
from logger import RoboLogger
from constant import LOGGER_OBJECT_DETECTOR_STARTUP, \
    LOGGER_OBJECT_DETECTOR_ERROR_HANDLER
from detector import ObjectDetector
import asyncio
import os
import sys
import argparse
import logging
import json
import traceback
import signal


log = RoboLogger.getLogger()
log.warning(LOGGER_OBJECT_DETECTOR_STARTUP,
            msg="Initial imports are completed.")


def main():
    """
    Full Jetson Detector Program
    """

    parser = argparse.ArgumentParser(description="Inference Program")
    parser.add_argument(
        '--config_file',
        help=f'json configuration file containint params '
             f'to initialize the jetson',
        type=str)
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
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(handle_exception)
        log.warning(LOGGER_OBJECT_DETECTOR_STARTUP,
                    msg='Launching runner.')
        signals = (signal.SIGINT, signal.SIGTERM)
        for s in signals:
            loop.add_signal_handler(s, lambda s=s: loop.create_task(
                objectDetector.graceful_shutdown(s)))
        objectDetector = ObjectDetector(robotJetsonConfiguration, loop)
        loop.create_task(objectDetector.run())
        loop.run_forever()

    except SystemExit:
        log.info(LOGGER_OBJECT_DETECTOR_STARTUP, 'Caught SystemExit...')
    except Exception:
        log.critical(LOGGER_OBJECT_DETECTOR_STARTUP,
                     'System Crash : {}'.format(traceback.print_exc()))
    finally:
        loop.close()
        logging.shutdown()


def handle_exception(loop, context):
    # context["message"] will always be there; but context["exception"] may not
    msg = context.get("exception", context["message"])
    log.error(
        LOGGER_OBJECT_DETECTOR_ERROR_HANDLER,
        f'Caught exception: {msg}')
    log.critical(
        LOGGER_OBJECT_DETECTOR_ERROR_HANDLER,
        f'Calling graceful_shutdown from exception handler.')
    loop.create_task(objectDetector.graceful_shutdown())


if __name__ == "__main__":
    main()
    log.info(LOGGER_OBJECT_DETECTOR_STARTUP, "Done")
    try:
        sys.exit(0)
    except SystemExit:
        log.info(LOGGER_OBJECT_DETECTOR_STARTUP, 'Exiting startup.')
