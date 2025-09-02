import time
from collections import deque

class CvFpsCalc(object):
    def __init__(self, buffer_len=10):
        self.start_tick = time.time()
        self.fps_deque = deque(maxlen=buffer_len)

    def get(self):
        current_tick = time.time()
        diff = current_tick - self.start_tick
        self.start_tick = current_tick
        fps = 1 / diff if diff > 0 else 0
        self.fps_deque.append(fps)
        return round(sum(self.fps_deque) / len(self.fps_deque), 2)
