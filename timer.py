from time import time

class TimedInterval:

    def __init__(self, name=None):
        self.name = name
        self.duration = None
        self.start_time = None
        self.subintervals = []
        self.data = {}

    def subinterval(self, name=None):
        sub = TimedInterval(name)
        self.subintervals.append(sub)
        return sub

    def put(self, key, value):
        self.data[key] = value

    def start(self):
        self.start_time = time()

    def stop(self):
        if self.duration is not None:
            return

        for sub in self.subintervals:
            sub.stop()
        
        if self.start_time is not None:
            self.duration = time() - self.start_time

    def show(self, indent=0):
        name = self.name if self.name else "anonymous"

        if self.duration >= 1:
            duration = f"{self.duration:.3f}s"
        else:
            duration = f"{int(self.duration * 1000)}ms"

        print(" " * indent + f"{name} - {duration}")
        for sub in self.subintervals:
            sub.show(indent=indent + 2)

    def __enter__(self):
        self.start()

    def __exit__(self, *exc_info):
        self.stop()