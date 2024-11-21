import numpy as np

def nearest_points(row, col, ax, dg, size = (30,30)):

    # row & col: Neuron corrdinates on which we plane to make connections
    # ax: The number of elements to find along the horizontal and vertical axes.
    # dg: The number of elements to find along the diagonal axes.
    # size: The size of the 2D array (both rows and columns).
    
    points = []

    # Left
    if ax > 0:
        temp_row = row
        temp_col = col - 1
        while temp_col >= 0:
            points.append((temp_row, temp_col))
            if (col - temp_col) == ax:
                break
            temp_col -= 1

    # Right
    if ax > 0:
        temp_row = row
        temp_col = col + 1
        while temp_col <= size[1]-1:
            points.append((temp_row, temp_col))
            if (temp_col - col) == ax:
                break
            temp_col += 1

    # Top
    if ax > 0:
        temp_row = row - 1
        temp_col = col
        while temp_row >= 0:
            points.append((temp_row, temp_col))
            if (row - temp_row) == ax:
                break
            temp_row -= 1

    # Bottom
    if ax > 0:
        temp_row = row + 1
        temp_col = col
        while temp_row <= size[0]-1:
            points.append((temp_row, temp_col))
            if (temp_row - row) == ax:
                break
            temp_row += 1

    # Top-left diagonal
    if dg > 0:
        temp_row = row - 1
        temp_col = col - 1
        while temp_row >= 0 and temp_col >= 0:
            points.append((temp_row, temp_col))
            if (row - temp_row) == dg:
                break
            temp_row -= 1
            temp_col -= 1

    # Top-right diagonal
    if dg > 0:
        temp_row = row - 1
        temp_col = col + 1
        while temp_row >= 0 and temp_col <= size[1]-1:
            points.append((temp_row, temp_col))
            if (row - temp_row) == dg:
                break
            temp_row -= 1
            temp_col += 1

    # Bottom-left diagonal
    if dg > 0:
        temp_row = row + 1
        temp_col = col - 1
        while temp_row <= size[0]-1 and temp_col >= 0:
            points.append((temp_row, temp_col))
            if (temp_row - row) == dg:
                break
            temp_row += 1
            temp_col -= 1

    # Bottom-right diagonal
    if dg > 0:
        temp_row = row + 1
        temp_col = col + 1
        while temp_row <= size[0]-1 and temp_col <= size[1]-1:
            points.append((temp_row, temp_col))
            if (temp_row - row) == dg:
                break
            temp_row += 1
            temp_col += 1

    return points


def rectangle_points(row, col, x, y, size=(30, 30)):
    # Boundaries for the grid
    max_row, max_col = size

    # Calculating the boundaries considering the size limits
    top_row = max(0, row - y)
    left_col = max(0, col - x)

    bottom_row = min(max_row - 1, row + y)
    right_col = min(max_col - 1, col + x)

  
    # Collecting all coordinates inside the rectangle
    points = []
    for r in range(top_row, bottom_row + 1):
        for c in range(left_col, right_col + 1):
            points.append((r, c))
    
    return points