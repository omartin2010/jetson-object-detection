import signal
import time
# import debugpy


def receiveSignal(signalNumber, frame):
    print('Received:', signalNumber)
    with open('sig.txt', 'w') as f:
        f.write(f'signal = {signalNumber}')
    time.sleep(4)
    with open('sig2.txt', 'w') as f:
        f.write(f'signal was able to wait 4 seconds...')
    return


if __name__ == '__main__':
    # register the signals to be caught
    signal.signal(signal.SIGHUP, receiveSignal)
    signal.signal(signal.SIGINT, receiveSignal)
    signal.signal(signal.SIGQUIT, receiveSignal)
    signal.signal(signal.SIGILL, receiveSignal)
    signal.signal(signal.SIGTRAP, receiveSignal)
    signal.signal(signal.SIGABRT, receiveSignal)
    signal.signal(signal.SIGBUS, receiveSignal)
    signal.signal(signal.SIGFPE, receiveSignal)
    # signal.signal(signal.SIGKILL, receiveSignal)
    signal.signal(signal.SIGUSR1, receiveSignal)
    signal.signal(signal.SIGSEGV, receiveSignal)
    signal.signal(signal.SIGUSR2, receiveSignal)
    signal.signal(signal.SIGPIPE, receiveSignal)
    signal.signal(signal.SIGALRM, receiveSignal)
    signal.signal(signal.SIGTERM, receiveSignal)
    i = 0
    while True:
        time.sleep(3)
        print(f'step...{i}')
        i += 1
