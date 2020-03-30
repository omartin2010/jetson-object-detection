from concurrent.futures import ProcessPoolExecutor
import time
import math


def work(x):
    result = math.factorial(x)
    return result


with ProcessPoolExecutor() as pool:
    arg = 1000000
    # Example of submitting work to the pool
    start_time = time.time()
    fr1 = pool.submit(work, arg)
    fr2 = pool.submit(work, arg)
    fr3 = pool.submit(work, arg)
    fr4 = pool.submit(work, arg)

    # Obtaining the result (blocks until done)
    r1 = fr1.result()
    r2 = fr2.result()
    r3 = fr3.result()
    r4 = fr4.result()
    print(f'Parallel compute time = {time.time() - start_time:.4f}s')

start_time = time.time()
r5 = work(arg)
r6 = work(arg)
r7 = work(arg)
r8 = work(arg)
print(f'Duration non parallel = {time.time() - start_time:.4f}')
