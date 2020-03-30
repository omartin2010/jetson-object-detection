from multiprocessing import Pool
import traceback


def f(x):
    return x * x


if __name__ == '__main__':
    # start 4 worker processes
    try:
        with Pool(processes=1) as pool:

            # print "[0, 1, 4,..., 81]"
            print(pool.map(f, range(10)))
            print("For the moment, the pool remains available for more work")

    except:
        print(traceback.print_exc())
    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")
