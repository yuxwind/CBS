# timer.py

import time

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self, info=''):
        self._start_time = None
        self.sum  = 0
        self.mean = 0
        self.N    = 0
        self.name = info

    def start(self):
        """Start a new timer"""
        #if self._start_time is not None:
        #    raise TimerError(f"Timer is running. Use .stop() to stop it")
        self._start_time = time.perf_counter()

    def stop(self, msg='', restart=True):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        if restart:
            self._start_time = time.perf_counter()
            self.N += 1
        else:
            self._start_time = None
        if msg == '':
            msg = f"{self.name}.{self.N}"
        #print(f"{msg:s} Elapsed time: {elapsed_time:0.4f} seconds")
        print(f"{msg:s} Elapsed time: {elapsed_time:0.8f} seconds")
        self.sum += elapsed_time
        self.mean = self.sum/self.N
        return elapsed_time

    def sum(self, ):
        return self.sum

    def mean(self,):
        return self.mean

    def count(self):
        return self.N

    def info(self, sinfo = ''):
        if sinfo == '':
            sinfo = self.name
        print(f"Time taken to {sinfo} is {self.sum:.2f} seconds: mean={self.mean:.2f}, N={self.N}")
