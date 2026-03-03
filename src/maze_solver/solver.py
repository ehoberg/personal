from typing import List
from util import all_directions

"""Maze solving algorithms. Currently implements depth-first search (DFS) both iteratively and recursively."""
class MazeSolver:
    
    """
    Args:
        maze: 2D list of ints representing the maze structure
        start: start position
        end: end position
    """
    def __init__(self, maze: List[List[int]], start: tuple[int, int], end: tuple[int, int]):
        self.maze = maze
        self.start = start
        self.end = end       

    """Return walkable neighbors (up, down, left, right)."""
    def __get_neighbors(self, pos: tuple[int, int]) -> List[tuple[int, int]]:
        neighbors = []
        for direction in all_directions:
            current_pos = tuple(a + b for a, b in zip(pos, direction))
            # Check for 0 values and out of bounds
            # Finally check if the space is walkable (not 1)
            if ( 0 <= current_pos[1] < len(self.maze) and
                 0 <= current_pos[0] < len(self.maze[0]) and
                 self.maze[current_pos] != 1):
                neighbors.append(current_pos)           
        return neighbors
    

    """Iterative depth-first search to find a path."""
    def depth_first(self) -> List[tuple[int, int]]:
        """Depth-first search to find a path"""
        stack = [self.start]
        visited = set()
        came_from = {self.start: None}

        while stack:
            current = stack.pop()
            if current == self.end:
                break
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.__get_neighbors(current):
                if neighbor not in visited:
                    stack.append(neighbor)
                    came_from[neighbor] = current

        # Reconstruct path
        path = []
        node = self.end
        while node:
            path.append(node)
            node = came_from.get(node)

        # Invert the path to get it from the start to end.
        return path[::-1]
    
    """Depth-first search to find a path"""
    def depth_first_recursive(self, visited, node) -> List[tuple[int, int]]:               
        # End reached, we are done
        if node == self.end:
            return [node]
       
        for neighbor in self.__get_neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                path = self.depth_first_recursive(visited, neighbor)

                # If the last node in the path is the start, we have found a valid path and can return it
                if path and path[-1] == self.start:                  
                    return path
                # Still building up the path, add the current node to the path and return it
                elif path:
                    return path + [node]
                    
        return []
