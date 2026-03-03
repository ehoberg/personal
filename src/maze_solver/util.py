import contextlib
import time

"""Utility classes and functions for maze generation and solving."""
class Direction:
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

all_directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]


"""Timer decorator and context manager for measuring execution time."""
class Timer(contextlib.ContextDecorator):
    
    # Usage: @Timer() decorator or 'with Timer('name'):' context manager
    def __init__(self, name: str = 'Timer'):
        self.name = name

    def __enter__(self):    
        self.start = time.perf_counter()

    def __exit__(self, _, __, ___):       
        print(f'{self.name}: {time.perf_counter() - self.start:.6f}s')


