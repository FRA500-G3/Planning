import heapq
import pygame
import numpy as np
from math import sin, cos, tan, radians, degrees, sqrt, pi
import yaml

# Load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


class Constants:
    GOAL_TOLERANCE = config["goal_tolerance"]
    SCREEN_SIZE = config["screen_size"]
    GRID_SIZE = config["grid_size"]
    VISION_RADIUS = config["vision_radius"]
    VEHICLE_WIDTH = config["vehicle_width"]
    VEHICLE_LENGTH = config["vehicle_length"]
    WHEEL_BASE = config["wheel_base"]
    MAX_STEERING_ANGLE = config["max_steering_angle"]
    STEP_SIZE = config["step_size"]
    GRID_SCALE = SCREEN_SIZE // GRID_SIZE
    TURNING_RADIUS = WHEEL_BASE / sin(radians(MAX_STEERING_ANGLE))


class Node:
    def __init__(self, x, y, theta, cost, parent=None, direction=1):
        self.x = x
        self.y = y
        self.theta = theta  # Orientation in degrees
        self.cost = cost
        self.parent = parent
        self.direction = direction  # 1 for forward, -1 for reverse

    def __lt__(self, other):
        return self.cost < other.cost


class HybridAStar:
    def __init__(self, obstacles, bounds, angle_resolution, resolution):
        self.obstacles = obstacles
        self.bounds = bounds
        self.angle_resolution = angle_resolution
        self.resolution = resolution

        self.motion_primitives = self.generate_motion_primitives()

        self.grid_size_phi = int(360 / self.angle_resolution)
        self.map_width = int((bounds[1] - bounds[0]) / self.resolution)
        self.map_height = int((bounds[3] - bounds[2]) / self.resolution)

        self.obstacle_map = self.create_obstacle_map()

        # Use a 3D boolean array for visited states
        self.visited = np.zeros(
            (self.map_width, self.map_height, self.grid_size_phi), dtype=bool
        )

        self.open_set = []
        self.path = []
        self.search_tree_edges = []

        # Precompute trigonometric values for all headings
        self.sin_table = np.zeros(self.grid_size_phi)
        self.cos_table = np.zeros(self.grid_size_phi)
        for i in range(self.grid_size_phi):
            angle_deg = i * self.angle_resolution
            angle_rad = radians(angle_deg)
            self.sin_table[i] = sin(angle_rad)
            self.cos_table[i] = cos(angle_rad)

        # State variables for threading
        self.searching = False
        self.search_done = False
        self.found = False

    def create_obstacle_map(self):
        obstacle_map = np.zeros((self.map_width, self.map_height), dtype=bool)
        for obs in self.obstacles:
            x_index = int((obs[0] - self.bounds[0]) / self.resolution)
            y_index = int((obs[1] - self.bounds[2]) / self.resolution)
            if 0 <= x_index < self.map_width and 0 <= y_index < self.map_height:
                obstacle_map[x_index, y_index] = True
        return obstacle_map

    def generate_motion_primitives(self):
        motions = []
        max_steering_angle = Constants.MAX_STEERING_ANGLE
        num_angles = int(max_steering_angle / self.angle_resolution)
        steering_angles = [
            i * self.angle_resolution for i in range(-num_angles, num_angles + 1)
        ]
        # Precompute tan for each steering angle to avoid repeated tan computations
        self.tan_values = {
            sa: tan(radians(sa)) if abs(sa) > 1e-6 else None for sa in steering_angles
        }

        for steering_angle in steering_angles:
            for direction in [1, -1]:  # Forward and reverse
                motions.append((steering_angle, direction))
        return motions

    def get_state_key(self, node):
        x_index = int(round((node.x - self.bounds[0]) / self.resolution))
        y_index = int(round((node.y - self.bounds[2]) / self.resolution))
        theta_index = int(node.theta / self.angle_resolution) % self.grid_size_phi
        return (x_index, y_index, theta_index)

    def is_valid(self, node):
        # Check bounds
        if not (
            self.bounds[0] <= node.x < self.bounds[1]
            and self.bounds[2] <= node.y < self.bounds[3]
        ):
            return False

        # Check collision with obstacles using the vehicle's footprint
        vehicle_corners = self.get_vehicle_corners(node)
        for corner in vehicle_corners:
            x_index = int(round((corner[0] - self.bounds[0]) / self.resolution))
            y_index = int(round((corner[1] - self.bounds[2]) / self.resolution))
            if (
                x_index < 0
                or x_index >= self.map_width
                or y_index < 0
                or y_index >= self.map_height
                or self.obstacle_map[x_index, y_index]
            ):
                return False

        # Optionally do a line collision check (can be simplified or skipped if expensive)
        if node.parent:
            if not self.line_collision_check(node.parent, node):
                return False
        return True

    def get_vehicle_corners(self, node):
        w = Constants.VEHICLE_WIDTH / 2.0
        l = Constants.VEHICLE_LENGTH / 2.0

        theta_index = int(node.theta / self.angle_resolution) % self.grid_size_phi
        cos_theta = self.cos_table[theta_index]
        sin_theta = self.sin_table[theta_index]

        corners = []
        # Compute corners using precomputed sin and cos
        for dx, dy in [(l, w), (l, -w), (-l, w), (-l, -w)]:
            x = node.x + dx * cos_theta - dy * sin_theta
            y = node.y + dx * sin_theta + dy * cos_theta
            corners.append((x, y))
        return corners

    def line_collision_check(self, node1, node2):
        x1, y1 = node1.x, node1.y
        x2, y2 = node2.x, node2.y
        dx = x2 - x1
        dy = y2 - y1
        distance = sqrt(dx**2 + dy**2)
        steps = int(distance / (self.resolution / 2))
        for i in range(steps + 1):
            t = i / max(steps, 1)
            x = x1 + t * dx
            y = y1 + t * dy
            x_index = int(round((x - self.bounds[0]) / self.resolution))
            y_index = int(round((y - self.bounds[2]) / self.resolution))
            if (
                x_index < 0
                or x_index >= self.map_width
                or y_index < 0
                or y_index >= self.map_height
                or self.obstacle_map[x_index, y_index]
            ):
                return False
        return True

    def heuristic(self, node):
        dx = self.goal.x - node.x
        dy = self.goal.y - node.y
        distance = sqrt(dx**2 + dy**2)
        return distance * (1.0 + 1e-3)

    def is_goal_reached(self, node):
        dx = self.goal.x - node.x
        dy = self.goal.y - node.y
        distance = sqrt(dx**2 + dy**2)
        dtheta = abs(self.goal.theta - node.theta) % 360
        if distance < Constants.GOAL_TOLERANCE and dtheta < self.angle_resolution:
            return True
        return False

    def create_successor(self, current, motion):
        steering_angle_deg, direction = motion
        phi = radians(steering_angle_deg)  # Steering angle in radians
        theta_rad = radians(current.theta)
        ds = direction * Constants.STEP_SIZE

        if abs(phi) < 1e-6:  # Straight line
            x = current.x + ds * cos(theta_rad)
            y = current.y + ds * sin(theta_rad)
            theta = current.theta
        else:
            # Use precomputed tan if available
            turning_radius = Constants.WHEEL_BASE / self.tan_values[steering_angle_deg]
            delta_theta_rad = ds / turning_radius
            theta_rad_next = theta_rad + delta_theta_rad
            x = current.x + turning_radius * (sin(theta_rad_next) - sin(theta_rad))
            y = current.y - turning_radius * (cos(theta_rad_next) - cos(theta_rad))
            theta = degrees(theta_rad_next) % 360

        cost = current.cost + Constants.STEP_SIZE
        successor = Node(x, y, theta, cost, current, direction)
        return successor

    def calculate_cost(self, current, successor):
        # Add extra cost for reversing or switching directions
        cost = 0
        if successor.direction != current.direction:
            if successor.direction == -1:
                cost += 10
        if successor.direction == -1:
            cost += 5
        return cost

    def reconstruct_path(self, node):
        path = []
        while node:
            path.append(node)
            node = node.parent
        path.reverse()
        return path

    def search(self, start):
        self.searching = True
        # Clear visited array for a new search
        self.visited.fill(False)

        # Push the start node into open_set
        heapq.heappush(self.open_set, (self.heuristic(start), start))
        max_iterations = 1000000
        iterations = 0

        while self.open_set and iterations < max_iterations:
            iterations += 1
            _, current = heapq.heappop(self.open_set)

            x_i, y_i, t_i = self.get_state_key(current)
            if self.visited[x_i, y_i, t_i]:
                continue

            self.visited[x_i, y_i, t_i] = True

            if self.is_goal_reached(current):
                self.path = self.reconstruct_path(current)
                self.found = True
                self.search_done = True
                self.searching = False
                return True

            for motion in self.motion_primitives:
                successor = self.create_successor(current, motion)
                x_s, y_s, t_s = self.get_state_key(successor)
                # Quickly check visited and bounds before is_valid to save time
                if (
                    0 <= x_s < self.map_width
                    and 0 <= y_s < self.map_height
                    and not self.visited[x_s, y_s, t_s]
                    and self.is_valid(successor)
                ):
                    total_cost = (
                        successor.cost
                        + self.heuristic(successor)
                        + self.calculate_cost(current, successor)
                    )
                    heapq.heappush(self.open_set, (total_cost, successor))
                    self.search_tree_edges.append((current, successor))

        return False
