from maze import Maze
from util import Timer
from solver import MazeSolver
import sys

def main():
    """Main entry point for the maze solver application."""
    # 500 x 500 its kinda iffy (it sometimes work), it will reach the recusion limit and crashes.
    maze = Maze(15, 15)


    # Guess we reach a recusion limit pretty quickly when making a maze of 500x500 =)
    # increase it slightly
    sys.setrecursionlimit(50000)  

    print("Initial Maze")    
    maze.generate_maze()
    print("-----------------------------------")
    maze.display()
    print("-----------------------------------")


    solver = MazeSolver(maze.grid, maze.start_pos, maze.end_pos)
    with Timer("Iterative DFS"):
        path = solver.depth_first()

    with Timer("Recursive DFS"):
        path_recursive = solver.depth_first_recursive(set(), maze.start_pos)
        path_recursive = path_recursive[::-1] if path_recursive else []
    
    print("---------------Iterative DFS--------------------")
    maze.display_with_path(path)
    print("---------------Initial Maze---------------------")
    maze.display()
    print("---------------Recursive DFS--------------------")
    maze.display_with_path(path_recursive)

    print(f"Iterative vs Recursive DFS: {'Same' if path == path_recursive else 'Different'}")


    '''
    Sample output for a 15x15 maze
    Iterative DFS: 0.000120s
    Recursive DFS: 0.000333s
    ---------------Iterative DFS--------------------
    xxxxxxxxxxxxxxx
    xB****x x     x
    x xxx*x x xxx x
    x x***x   x   x
    xxx*xxxxxxx xxx
    x x*x       xxx
    x x*x x xxxxxxx
    x x*x x       x
    x x*x xxxxxxx x
    x x*x x   x   x
    x x*x x x x x x
    x***x   x x x x
    x*xxxxxxxxx x x
    x************Ex
    xxxxxxxxxxxxxxx
    ---------------Initial Maze---------------------
    xxxxxxxxxxxxxxx
    xB    x x     x
    x xxx x x xxx x
    x x   x   x   x
    xxx xxxxxxx xxx
    x x x       xxx
    x x x x xxxxxxx
    x x x x       x
    x x x xxxxxxx x
    x x x x   x   x
    x x x x x x x x
    x   x   x x x x
    x xxxxxxxxx x x
    x            Ex
    xxxxxxxxxxxxxxx
    ---------------Recursive DFS--------------------
    xxxxxxxxxxxxxxx
    xB****x x     x
    x xxx*x x xxx x
    x x***x   x   x
    xxx*xxxxxxx xxx
    x x*x       xxx
    x x*x x xxxxxxx
    x x*x x       x
    x x*x xxxxxxx x
    x x*x x   x***x
    x x*x x x x*x*x
    x***x   x x*x*x
    x*xxxxxxxxx*x*x
    x*********** Ex
    xxxxxxxxxxxxxxx
    
    '''

if __name__ == "__main__":
    main()