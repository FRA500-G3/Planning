import pygame
import numpy as np
import heapq
from math import sin, cos, tan, radians, degrees, sqrt, pi


# Constants
class Constants:
    RESOLUTION = 1
    ANGLE_RESOLUTION = 15
    GOAL_TOLERANCE = 1
    SCREEN_SIZE = 600
    GRID_SIZE = 15
    VISION_RADIUS = 2
    VEHICLE_WIDTH = 0.5  # Width of the vehicle
    VEHICLE_LENGTH = 1  # Length of the vehicle
    WHEEL_BASE = 1.0  # Distance between front and rear axles
    MAX_STEERING_ANGLE = 30  # Degrees
    TURNING_RADIUS = WHEEL_BASE / sin(radians(MAX_STEERING_ANGLE))
    STEP_SIZE = 1
    GRID_SCALE = SCREEN_SIZE // GRID_SIZE


# Node class
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
    def __init__(self, start, goal, obstacles, bounds):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.bounds = bounds
        self.open_set = []
        self.closed_set = set()
        self.motion_primitives = self.generate_motion_primitives()
        self.path = []
        self.search_tree_edges = []
        self.grid_size_phi = int(360 / Constants.ANGLE_RESOLUTION)
        self.map_width = int((bounds[1] - bounds[0]) / Constants.RESOLUTION)
        self.map_height = int((bounds[3] - bounds[2]) / Constants.RESOLUTION)
        self.obstacle_map = self.create_obstacle_map()

    def create_obstacle_map(self):
        obstacle_map = np.zeros((self.map_width, self.map_height), dtype=bool)
        for obs in self.obstacles:
            x_index = int(obs[0] - self.bounds[0])
            y_index = int(obs[1] - self.bounds[2])
            if 0 <= x_index < self.map_width and 0 <= y_index < self.map_height:
                obstacle_map[x_index, y_index] = True
        return obstacle_map

    def generate_motion_primitives(self):
        motions = []
        max_steering_angle = Constants.MAX_STEERING_ANGLE
        num_angles = int(max_steering_angle / Constants.ANGLE_RESOLUTION)
        steering_angles = [
            i * Constants.ANGLE_RESOLUTION for i in range(-num_angles, num_angles + 1)
        ]
        for steering_angle in steering_angles:
            for direction in [1, -1]:  # Forward and reverse
                motions.append((steering_angle, direction))
        return motions

    def get_state_key(self, node):
        x_index = int(round((node.x - self.bounds[0]) / Constants.RESOLUTION))
        y_index = int(round((node.y - self.bounds[2]) / Constants.RESOLUTION))
        theta_index = int(node.theta / Constants.ANGLE_RESOLUTION) % self.grid_size_phi
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
            x_index = int(round((corner[0] - self.bounds[0]) / Constants.RESOLUTION))
            y_index = int(round((corner[1] - self.bounds[2]) / Constants.RESOLUTION))
            if (
                x_index < 0
                or x_index >= self.map_width
                or y_index < 0
                or y_index >= self.map_height
                or self.obstacle_map[x_index, y_index]
            ):
                return False
        # Line collision check between parent and current node
        if node.parent:
            if not self.line_collision_check(node.parent, node):
                return False
        return True

    def get_vehicle_corners(self, node):
        w = Constants.VEHICLE_WIDTH / 2.0
        l = Constants.VEHICLE_LENGTH / 2.0
        theta_rad = radians(node.theta)
        cos_theta = cos(theta_rad)
        sin_theta = sin(theta_rad)
        corners = []
        for dx, dy in [
            (l, w),
            (l, -w),
            (-l, w),
            (-l, -w),
        ]:
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
        steps = int(distance / (Constants.RESOLUTION / 2))
        for i in range(steps + 1):
            t = i / max(steps, 1)
            x = x1 + t * dx
            y = y1 + t * dy
            x_index = int(round((x - self.bounds[0]) / Constants.RESOLUTION))
            y_index = int(round((y - self.bounds[2]) / Constants.RESOLUTION))
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
        # Tie-breaker to improve performance
        return distance * (1.0 + 1e-3)

    def is_goal_reached(self, node):
        dx = self.goal.x - node.x
        dy = self.goal.y - node.y
        distance = sqrt(dx**2 + dy**2)
        dtheta = abs(self.goal.theta - node.theta) % 360
        if distance < Constants.GOAL_TOLERANCE and dtheta < Constants.ANGLE_RESOLUTION:
            return True
        return False

    def create_successor(self, current, motion):
        steering_angle_deg, direction = motion
        phi = radians(steering_angle_deg)  # Steering angle in radians
        theta_rad = radians(current.theta)  # Current heading in radians
        ds = direction * Constants.STEP_SIZE  # Distance to move
        if abs(phi) < 1e-6:  # Straight movement
            x = current.x + ds * cos(theta_rad)
            y = current.y + ds * sin(theta_rad)
            theta = current.theta  # No change in heading
        else:
            turning_radius = Constants.WHEEL_BASE / tan(phi)
            delta_theta_rad = ds / turning_radius  # Change in heading in radians
            theta_rad_next = theta_rad + delta_theta_rad
            x = current.x + turning_radius * (sin(theta_rad_next) - sin(theta_rad))
            y = current.y - turning_radius * (cos(theta_rad_next) - cos(theta_rad))
            theta = degrees(theta_rad_next) % 360
        cost = current.cost + Constants.STEP_SIZE
        successor = Node(x, y, theta, cost, current, direction)
        return successor

    def calculate_cost(self, current, successor):
        # Implement cost calculation, including penalties for reversing, steering changes, etc.
        cost = Constants.STEP_SIZE
        if successor.direction != current.direction:
            cost += Constants.STEP_SIZE  # Penalty for changing direction
        if successor.theta != current.theta:
            cost += Constants.STEP_SIZE  # Penalty for steering
        if successor.direction == -1:
            cost += 1000000000000000000000000
        return cost

    def reconstruct_path(self, node):
        path = []
        while node:
            path.append(node)
            node = node.parent
        path.reverse()
        return path

    def search(self):
        heapq.heappush(self.open_set, (self.heuristic(self.start), self.start))
        max_iterations = 10000
        iterations = 0

        while self.open_set and iterations < max_iterations:
            iterations += 1
            _, current = heapq.heappop(self.open_set)

            current_key = self.get_state_key(current)
            if current_key in self.closed_set:
                continue
            self.closed_set.add(current_key)

            if self.is_goal_reached(current):
                self.path = self.reconstruct_path(current)
                return True

            for motion in self.motion_primitives:
                successor = self.create_successor(current, motion)
                if not self.is_valid(successor):
                    continue
                successor_key = self.get_state_key(successor)
                if successor_key in self.closed_set:
                    continue
                total_cost = successor.cost + self.heuristic(successor)
                heapq.heappush(self.open_set, (total_cost, successor))
                self.search_tree_edges.append((current, successor))
        return False


def dynamic_path_planning(start, goal, obstacles, bounds):
    pygame.init()
    screen = pygame.display.set_mode((Constants.SCREEN_SIZE, Constants.SCREEN_SIZE))
    clock = pygame.time.Clock()
    running = True

    grid_scale = Constants.GRID_SCALE
    robot_pos = [start.x, start.y]
    robot_theta = start.theta

    planner = HybridAStar(start, goal, obstacles, bounds)
    found = planner.search()
    if not found:
        print("No path found!")
    path = planner.path
    path_index = 0
    last_replan_time = 0  # To track time since last replanning

    # Define the robot size in pixels
    vehicle_width_px = Constants.VEHICLE_WIDTH * grid_scale
    vehicle_length_px = Constants.VEHICLE_LENGTH * grid_scale

    previous_positions = []
    last_recorded_pos = [robot_pos[0], robot_pos[1], robot_theta]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:  # Handle mouse click events
                # Get mouse position
                mouse_x, mouse_y = pygame.mouse.get_pos()

                # Convert mouse position to grid coordinates
                grid_x = mouse_x // grid_scale
                grid_y = mouse_y // grid_scale

                # Add obstacle to the list if not already present
                if (grid_x, grid_y) not in obstacles:
                    obstacles.append((grid_x, grid_y))
                    # Replan
                    planner = HybridAStar(
                        Node(robot_pos[0], robot_pos[1], robot_theta, 0),
                        goal,
                        obstacles,
                        bounds,
                    )
                    found = planner.search()
                    if not found:
                        print("No path found!")
                    path = planner.path
                    path_index = 0

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                # Set new goal
                mouse_x, mouse_y = pygame.mouse.get_pos()
                grid_x = mouse_x / grid_scale
                grid_y = mouse_y / grid_scale
                goal = Node(grid_x, grid_y, 0, 0)
                # Replan
                planner = HybridAStar(
                    Node(robot_pos[0], robot_pos[1], robot_theta, 0),
                    goal,
                    obstacles,
                    bounds,
                )
                found = planner.search()
                if not found:
                    print("No path found!")
                path = planner.path
                path_index = 0

        screen.fill((255, 255, 255))

        # Draw grid
        for x in range(0, Constants.SCREEN_SIZE, grid_scale):
            pygame.draw.line(
                screen, (200, 200, 200), (x, 0), (x, Constants.SCREEN_SIZE)
            )
        for y in range(0, Constants.SCREEN_SIZE, grid_scale):
            pygame.draw.line(
                screen, (200, 200, 200), (0, y), (Constants.SCREEN_SIZE, y)
            )

        # Draw obstacles
        for obs in obstacles:
            pygame.draw.rect(
                screen,
                (255, 0, 0),  # Red color for obstacles
                (
                    obs[0] * grid_scale,  # Top-left x position of the grid cell
                    obs[1] * grid_scale,  # Top-left y position of the grid cell
                    grid_scale,  # Width of the grid cell
                    grid_scale,  # Height of the grid cell
                ),
            )

        # Draw start and goal
        start_rect = pygame.Rect(
            int(start.x * grid_scale - vehicle_length_px / 2),
            int(start.y * grid_scale - vehicle_width_px / 2),
            vehicle_length_px,
            vehicle_width_px,
        )
        pygame.draw.rect(screen, (0, 255, 0), start_rect, 1)

        goal_center = (
            int(goal.x * grid_scale),  # X-coordinate of the goal's center
            int(goal.y * grid_scale),  # Y-coordinate of the goal's center
        )

        goal_radius = int(Constants.GOAL_TOLERANCE * grid_scale)

        # Draw the goal circle on the screen
        pygame.draw.circle(
            screen, (255, 0, 255), goal_center, goal_radius, 1
        )  # 1 is the thickness

        # Draw the search tree (yellow lines)
        for edge in planner.search_tree_edges:
            parent, child = edge
            pygame.draw.line(
                screen,
                (255, 200, 0),
                (int(parent.x * grid_scale), int(parent.y * grid_scale)),
                (int(child.x * grid_scale), int(child.y * grid_scale)),
                1,
            )

        # Draw robot vision
        pygame.draw.circle(
            screen,
            (0, 255, 255),
            (int(robot_pos[0] * grid_scale), int(robot_pos[1] * grid_scale)),
            Constants.VISION_RADIUS * grid_scale,
            1,
        )

        # Move the robot along the path
        if path and path_index < len(path):
            target_node = path[path_index]
            delta_x = target_node.x - robot_pos[0]
            delta_y = target_node.y - robot_pos[1]
            delta_theta = (
                target_node.theta - robot_theta + 180 + 360
            ) % 360 - 180  # Shortest angle

            robot_pos[0] += delta_x * 0.1
            robot_pos[1] += delta_y * 0.1
            robot_theta += delta_theta * 0.1

            if sqrt(delta_x**2 + delta_y**2) < 0.1 and abs(delta_theta) < 1:
                path_index += 1

            # Record previous position if moved enough
            if (
                sqrt(
                    (robot_pos[0] - last_recorded_pos[0]) ** 2
                    + (robot_pos[1] - last_recorded_pos[1]) ** 2
                )
                >= 0.5
            ):
                previous_positions.append(
                    (last_recorded_pos[0], last_recorded_pos[1], robot_theta)
                )
                last_recorded_pos = [robot_pos[0], robot_pos[1], robot_theta]

        # Draw the previous positions as thin-lined rotated rectangles
        for pos_x, pos_y, pos_theta in previous_positions:
            # Create a surface for the vehicle
            vehicle_surface_prev = pygame.Surface(
                (vehicle_length_px, vehicle_width_px), pygame.SRCALPHA
            )
            pygame.draw.rect(
                vehicle_surface_prev,
                (0, 128, 255),
                (0, 0, vehicle_length_px, vehicle_width_px),
                1,  # Outline only
            )
            # Rotate the surface
            rotated_vehicle_prev = pygame.transform.rotate(
                vehicle_surface_prev, -pos_theta
            )
            rotated_rect_prev = rotated_vehicle_prev.get_rect(
                center=(pos_x * grid_scale, pos_y * grid_scale)
            )
            screen.blit(rotated_vehicle_prev, rotated_rect_prev.topleft)

        # Draw the robot as a rotated rectangle
        vehicle_surface = pygame.Surface(
            (vehicle_length_px, vehicle_width_px), pygame.SRCALPHA
        )
        pygame.draw.rect(
            vehicle_surface,
            (0, 128, 255),
            (0, 0, vehicle_length_px, vehicle_width_px),
        )

        # Rotate the vehicle surface according to the robot's orientation
        rotated_vehicle = pygame.transform.rotate(vehicle_surface, -robot_theta)
        rotated_rect = rotated_vehicle.get_rect(
            center=(robot_pos[0] * grid_scale, robot_pos[1] * grid_scale)
        )

        # Blit the rotated vehicle onto the screen
        screen.blit(rotated_vehicle, rotated_rect.topleft)

        # Draw the predicted path (blue line)
        if path:
            for i in range(len(path) - 1):
                pygame.draw.line(
                    screen,
                    (0, 0, 255),
                    (int(path[i].x * grid_scale), int(path[i].y * grid_scale)),
                    (
                        int(path[i + 1].x * grid_scale),
                        int(path[i + 1].y * grid_scale),
                    ),
                    2,
                )

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


# Example usage
start = Node(1, 1, 0, 0)
goal = Node(12, 12, 0, 0)
obstacles = [
    (5, 5),
    (6, 5),
    (7, 5),
    (12, 5),
    (13, 5),
    (5, 6),
    (5, 7),
    (8, 5),
    (9, 5),
    (10, 5),
    (11, 5),
]
bounds = (0, 15, 0, 15)

# Run dynamic path planning
dynamic_path_planning(start, goal, obstacles, bounds)
