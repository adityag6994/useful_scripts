import numpy as np
import random
import copy

# class node to hold value of point, with cost
class node:
    def __init__(self, pt, cost_a, cost_b):
        self.pt = pt
        self.cost_a = cost_a  # movement cost
        self.cost_b = cost_b  # distance from goal
        self.total = self.cost_a + self.cost_b

    def __lt__(self, other):
        return self.total < other.total

random.seed(1)

robot_map = np.ndarray((20, 20))
robot_map.fill(1)

# random obstacle
# obstacle_y = random.sample(range(0,19), 14)
# obstacle_x = random.sample(range(0,19), 14)

# pre-defined obstacle
obstacle_y = [4, 4, 4, 4,
              4, 5, 6, 7,
              8, 9, 10, 11,
              12, 13, 14, 15,
              16, 16, 16, 16,
              16, 16]
obstacle_x = [8, 9, 10, 11,
              12, 13, 13, 13,
              13, 13, 13, 13,
              13, 13, 13, 13,
              13, 12, 11, 10,
              9, 8]

# activate obstacle
robot_map[obstacle_y, obstacle_x] = -1

# define robot cost map for visulaisation
robot_cost = np.ndarray((20, 20))
robot_cost.fill(0)

# size of roboto map
grid_y = robot_map.shape[0]
grid_x = robot_map.shape[1]

# start and end points
start = np.array([0, 0])
goal = np.array([18, 18])

# possible moves of robot
moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]

# list to store visited, and explored nodes
visited = []
openSet = []

# cost and total moves
cost_so_far = 0
total_moves = 0

# cost, per moves, (affects how next move would be taken by robot)
cost_per_move = 0.1

openSet.append(node(start, cost_per_move, np.linalg.norm(start - goal)))
current = openSet[0]

# loop until we have next point in openSet
while len(openSet):

    # if goal reached, exit the loop
    if(current.pt == goal).all():
        print("Goal Reached!")
        exit()

    # get minimum from openSet
    openSet.sort()
    current = copy.deepcopy(openSet[0])

    #flag to check if current point, has valid neibhors or not
    step_flag = False

    # update robot cost for visulisation
    robot_cost[current.pt[0]][current.pt[1]] = current.cost_a + current.cost_b
    openSet.pop(0)
    for i, m in enumerate(moves):
        # current neibhor
        current_place = current.pt + m
        # check if within boundaries, and not already visited and not an obstacle
        if grid_y > current_place[0] >= 0 and \
                0 <= current_place[1] < grid_y and \
                robot_map[current_place[0]][[current_place[1]]] > 0 and \
                len([p for p in visited if (p.pt == current.pt).all()]) == 0:
            openSet.append(node(current_place,
                                cost_per_move * total_moves,
                                np.linalg.norm(current_place - goal)))
            step_flag = True

    if step_flag:
        # update stuff
        robot_map[current.pt[0]][current.pt[1]] = 3
        cost_so_far = cost_per_move*total_moves
        total_moves += 1
        visited.append(node(current.pt,
                            current.cost_a,
                            current.cost_b))
        print("current : ", current.pt, total_moves)
        step_flag = False

print("Cannot Reach the goal")
