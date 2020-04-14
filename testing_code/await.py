import asyncio
import sys
import signal
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s,%(msecs)d %(levelname)s: %(message)s", datefmt="%H:%M:%S")


class myobj(object):
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.shutdown = False
        self.loop = loop

    async def graceful_shutdown(self, s=None):
        try:
            if s is not None:
                logging.warning(f'Receiving signal {s.name}.')
            else:
                logging.warning(f'Shutting NOT via signal')
            tasks = [t for t in asyncio.Task.all_tasks(loop=self.loop) if t is not asyncio.Task.current_task()]
            # self.shutdown_task = asyncio.Task.current_task()
            logging.warning(f'Initiating cancellation of {len(tasks)} tasks...')
            [task.cancel() for task in tasks]
            logging.warning(f'Gaterhing out put of cancellation of {len(tasks)} tasks...')
            await asyncio.gather(*tasks, loop=self.loop, return_exceptions=True)
            logging.warning(f'Done cancelling tasks.')
        except:
            logging.warning('Problem cancelling task...')
        finally:
            logging.warning('Done graceful shutdown of subtasks')
            self.shutdown = True
            self.loop.stop()

    async def main(self):
        i = 0
        while not self.shutdown:
            await asyncio.sleep(1)
            print(f'Main runner... {i}')
            i += 1
        logging.warning('Main runner is over.')

    async def run(self):
        self.loop.create_task(self.main())
        self.loop.create_task(self.task_x())
        self.loop.create_task(self.task_y())
        logging.warning('Ending runner launching task.')

    async def task_x(self):
        logging.warning('Starting task X')
        i = 0
        while True:
            await asyncio.sleep(2)
            print(f'TASK X - Loop {i}')
            if i == 30:
                raise RuntimeError('BOOM X!')
            i += 1

    async def task_y(self):
        logging.warning('Starting task Y')
        i = 0
        while True:
            await asyncio.sleep(1.5)
            print(f'TASK Y - Loop {i}')
            if i == 32:
                raise RuntimeError('BOOM Y!')
            i += 1


def handle_exception(loop, context):
    # context["message"] will always be there; but context["exception"] may not
    msg = context.get("exception", context["message"])
    logging.warning(f'Caught exception: {msg}')
    logging.warning(f'Calling graceful_shutdown from exception handler.')
    loop.create_task(newobj.graceful_shutdown())


def main():
    try:
        global newobj
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(handle_exception)
        newobj = myobj(loop)
        signals = (signal.SIGINT, signal.SIGTERM)
        for s in signals:
            loop.add_signal_handler(s, lambda s=s: loop.create_task(
                newobj.graceful_shutdown(s)))
        loop.create_task(newobj.run())
        logging.warning(f'Starting event loop now.')
        loop.run_forever()
    finally:
        loop.close()
        logging.warning(f'object is Shutdown - Exiting program.')
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        logging.info('Exiting startup.')
