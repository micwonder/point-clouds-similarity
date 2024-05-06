def find_matching_robots(map, query):
    rows = len(map)
    cols = len(map[0])

    # Preprocess to calculate distances to nearest blockers in all four directions
    left = [[0] * cols for _ in range(rows)]
    right = [[0] * cols for _ in range(rows)]
    up = [[0] * cols for _ in range(rows)]
    down = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            if map[i][j] == "X":
                left[i][j] = 0
                up[i][j] = 0
            else:
                left[i][j] = left[i][j - 1] + 1 if j > 0 else 1
                up[i][j] = up[i - 1][j] + 1 if i > 0 else 1

    for i in range(rows - 1, -1, -1):
        for j in range(cols - 1, -1, -1):
            if map[i][j] == "X":
                right[i][j] = 0
                down[i][j] = 0
            else:
                right[i][j] = right[i][j + 1] + 1 if j < cols - 1 else 1
                down[i][j] = down[i + 1][j] + 1 if i < rows - 1 else 1

    ans = []

    for i in range(rows):
        for j in range(cols):
            if map[i][j] == "O":
                # Check if distances match the query
                if (
                    left[i][j] == query[0]
                    and right[i][j] == query[3]
                    and up[i][j] == query[1]
                    and down[i][j] == query[2]
                ):
                    ans.append([i, j])

    return ans


map = [
    ["O", "E", "E", "E", "X"],
    ["E", "O", "X", "X", "X"],
    ["E", "E", "E", "E", "E"],
    ["X", "E", "O", "E", "E"],
    ["X", "E", "X", "E", "X"],
]
query = [2, 2, 4, 1]
print(find_matching_robots(map, query))
