import numpy as np
from util import all_directions

"""Maze generator.

Maze uses a grid with integer cell values:
0: path, 1: wall, 2: start, 3: end, 4: path marker for display.

Generation uses a recursive carve algorithm (randomized depth-first).
"""
class Maze:

    symbol_map = {
        0: " ",
        1: "x",
        2: "B",
        3: "E",
        4: "*"
    }

    """Grid-based maze.

    Args:
        width: Desired grid width (minimum 3).
        height: Desired grid height (minimum 3).

    Attributes:
        grid: 2D numpy array of ints representing the maze structure.
        start_pos: (row, col) start cell.
        end_pos: (row, col) end cell.
    """
    def __init__(self, width: int, height: int):
        # Ensure minimum size of 3x3 to allow for paths and walls
        self.width = 3 if width < 3 else width
        self.height = 3 if height < 3 else height
        
        # 1 for walls, 0 for paths
        self.grid = np.ones((self.height, self.width), dtype=int)  

        # Top-left corner is the start, bottom-right corner is the end
        self.start_pos = (1, 1)
        self.end_pos = (self.width - 2, self.height - 2)        

    """Display the maze in the console using symbols."""
    def display(self, grid = None):
        if grid is None:
            grid = self.grid
        
        for row in grid:
            print("".join(self.symbol_map.get(cell, "?") for cell in row))

    """Display the maze with a path overlay."""
    def display_with_path(self, path):        
        clone = self.grid.copy()
        for entry in path:
            clone[entry] = 4 if clone[entry] == 0 else clone[entry]
        
        self.display(grid=clone)

    """Generate a maze using a recursive carving algorithm."""
    def generate_maze(self):
       
        # Mark start and end positions
        self.grid[self.start_pos] = 2 
        self.grid[self.end_pos] = 3  
        
        # Recusive function to walk the maze and fill it with paths
        def walk_and_fill_maze(pos = tuple[int, int]):
            
            self.grid[pos] = 0 if self.grid[pos] == 1 else self.grid[pos]  

            # Grab a random direction to walk in
            np.random.shuffle(all_directions)

            y,x = pos
            
            for dy, dx in all_directions:
                # look one extra step in the direction to see if it's a wall or if we can walk further
                neighbor_pos = (y + dy * 2, x + dx * 2)
                if 0 <= neighbor_pos[1] < self.width and 0 <= neighbor_pos[0] < self.height and self.grid[neighbor_pos] != 0:
                    self.grid[y + dy][x + dx] = 0 
                    walk_and_fill_maze(neighbor_pos)

        walk_and_fill_maze(self.start_pos)