from copy import deepcopy
import asyncio
import random
import traceback
import logging
logging.basicConfig(level=logging.DEBUG)


def custom_handler(loop, context):
    print('in custom handler...')
    if 'exception' in context:
        # print(context['exception'])
        raise context['exception']
        # print(context['message'])


async def dummy(i):
    print(f'Starting dummy task...{i}')
    # future.set_exception(Exception('Problem'))
    raise Exception(f'Problem! {i}')


async def dummy_2(i):
    print(f'starting dummy 2...{i}')
    raise Exception('Problem dummy 2-{i}')


async def dummy_forever(loop):
    i = 0
    tasks = []
    while True:
        try:
            print(f'test...{i}')
            await asyncio.sleep(1)
            if random.randint(1, 3) == 2:
                print(f'Attemtping to create new crashing task at i={i}...')
                # loop.create_task(dummy(i))
                tasks.append(loop.create_task(dummy(i)))
                loop.create_task(dummy_2(i))
            i += 1
        except:
            print("in dummy...")
            print(f'traceback = {traceback.print_exc()}')
            raise
        finally:
            # Check for crashes in tasks...
            new_tasks = []
            for task in tasks:
                if task.done() and not task.cancelled():
                    print(f'PROBLEM : {task.exception()}')
                else:
                    new_tasks.append(task)
            tasks = new_tasks


def test():
    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(custom_handler)
        loop.run_until_complete(dummy_forever(loop))
        # loop.run_forever()
    except Exception:
        print(f'in test - {traceback.print_exc()}')
    finally:
        loop.close()


test()
