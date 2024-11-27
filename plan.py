import pygame
import numpy as np
import heapq
from math import sin, cos, radians, sqrt


# Node class
class Node:
    def __init__(self, x, y, theta, cost, parent=None):
        self.x = x
        self.y = y
        self.theta = theta  # Orientation in degrees
        self.cost = cost
        self.parent = parent

    def __lt__(self, other):
        return self.cost < other.cost


# Constants
class Constants:
    RESOLUTION = 1
    ANGLE_RESOLUTION = 15
    GOAL_TOLERANCE = 1
    SCREEN_SIZE = 600
    GRID_SIZE = 15
    VISION_RADIUS = 2


# Vehicle dimensions (in grid units)
VEHICLE_WIDTH = 1  # Width of the vehicle
VEHICLE_HEIGHT = 0.5  # Height of the vehicle


# Heuristic
def heuristic(node, goal):
    return sqrt((goal.x - node.x) ** 2 + (goal.y - node.y) ** 2)


# Motion model
def generate_motion_model(step_size):
    motions = []
    for delta_theta in [-1, 0, 1]:
        for reverse in [1, -1]:
            motions.append(
                (delta_theta * Constants.ANGLE_RESOLUTION, reverse, step_size)
            )
    return motions


def is_valid(node, obstacles, bounds):
    # Check bounds
    if not (bounds[0] <= node.x < bounds[1] and bounds[2] <= node.y < bounds[3]):
        return False

    # Robot size in terms of grid units
    robot_half_width = VEHICLE_WIDTH / 2
    robot_half_height = VEHICLE_HEIGHT / 2

    # Check for collision with any obstacle
    for obs in obstacles:
        obs_x, obs_y = obs
        # Check if any part of the robot's bounding box overlaps with the obstacle's grid cell
        if (
            obs_x - robot_half_width <= node.x < obs_x + 1 + robot_half_width
            and obs_y - robot_half_height <= node.y < obs_y + 1 + robot_half_height
        ):
            return False

    return True


def discretize(value, resolution):
    return round(value / resolution) * resolution


def hybrid_a_star(start, goal, obstacles, bounds, robot_pos):
    open_set = []
    closed_set = set()
    heapq.heappush(open_set, (0 + heuristic(start, goal), start))
    search_tree_edges = []  # Keep track of the search tree edges

    # Define step sizes and thresholds
    coarse_step_size = 15 * Constants.RESOLUTION
    medium_step_size = 2 * Constants.RESOLUTION
    fine_step_size = 1 * Constants.RESOLUTION

    threshold_far = 15  # units
    threshold_near = 10  # units

    # Precompute motion models
    motion_model_coarse = generate_motion_model(coarse_step_size)
    motion_model_medium = generate_motion_model(medium_step_size)
    motion_model_fine = generate_motion_model(fine_step_size)

    while open_set:
        _, current = heapq.heappop(open_set)

        # Discretize the current node's position
        x_discrete = discretize(current.x, Constants.RESOLUTION)
        y_discrete = discretize(current.y, Constants.RESOLUTION)
        theta_discrete = (
            round(current.theta / Constants.ANGLE_RESOLUTION)
            * Constants.ANGLE_RESOLUTION
        ) % 360

        closed_set.add((x_discrete, y_discrete, theta_discrete))

        if heuristic(current, goal) < Constants.GOAL_TOLERANCE:
            path = []
            while current:
                path.append((current.x, current.y, current.theta))
                current = current.parent
            return path[::-1], search_tree_edges  # Return the path and the search tree

        # Compute distance to robot's position
        distance_to_robot = sqrt(
            (current.x - robot_pos[0]) ** 2 + (current.y - robot_pos[1]) ** 2
        )

        # Determine which motion model to use
        if distance_to_robot > threshold_far:
            motion_model = motion_model_coarse
        elif distance_to_robot > threshold_near:
            motion_model = motion_model_medium
        else:
            motion_model = motion_model_fine

        for motion in motion_model:
            delta_theta, reverse, step_size = motion
            theta_next = (current.theta + delta_theta) % 360
            x_next = current.x + reverse * cos(radians(theta_next)) * step_size
            y_next = current.y + reverse * sin(radians(theta_next)) * step_size
            cost_next = current.cost + step_size

            x_next_discrete = discretize(x_next, Constants.RESOLUTION)
            y_next_discrete = discretize(y_next, Constants.RESOLUTION)
            theta_next_discrete = (
                round(theta_next / Constants.ANGLE_RESOLUTION)
                * Constants.ANGLE_RESOLUTION
            ) % 360

            if (x_next_discrete, y_next_discrete, theta_next_discrete) in closed_set:
                continue

            next_node = Node(x_next, y_next, theta_next, cost_next, current)

            if not is_valid(next_node, obstacles, bounds):
                continue

            f_cost = cost_next + heuristic(next_node, goal)

            heapq.heappush(open_set, (f_cost, next_node))

            # Add the edge to the search tree
            search_tree_edges.append((current, next_node))

    return None, search_tree_edges


def dynamic_path_planning(start, goal, obstacles, bounds):
    pygame.init()
    screen = pygame.display.set_mode((Constants.SCREEN_SIZE, Constants.SCREEN_SIZE))
    clock = pygame.time.Clock()
    running = True

    grid_scale = Constants.SCREEN_SIZE // Constants.GRID_SIZE
    path_index = 0
    robot_pos = [start.x, start.y]
    robot_theta = start.theta
    path, search_tree_edges = hybrid_a_star(start, goal, obstacles, bounds, robot_pos)
    last_replan_time = 0  # To track time since last replanning

    # Define the robot size in pixels
    vehicle_width_px = VEHICLE_WIDTH * grid_scale
    vehicle_height_px = VEHICLE_HEIGHT * grid_scale

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

        # Draw start and goal as rectangles
        start_rect = pygame.Rect(
            int(start.x * grid_scale - vehicle_width_px / 2),
            int(start.y * grid_scale - vehicle_height_px / 2),
            vehicle_width_px,
            vehicle_height_px,
        )
        pygame.draw.rect(screen, (0, 255, 0), start_rect, 1)

        goal_center = (
            int(goal.x * grid_scale),  # X-coordinate of the goal's center
            int(goal.y * grid_scale),  # Y-coordinate of the goal's center
        )

        goal_radius = int(Constants.GOAL_TOLERANCE * grid_scale)

        # Draw the circle on the screen
        pygame.draw.circle(
            screen, (255, 0, 255), goal_center, goal_radius, 1
        )  # 1 is the thickness

        # Draw the search tree (yellow lines)
        for edge in search_tree_edges:
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

        # Detect obstacles in vision radius
        visible_obstacles = [
            obs
            for obs in obstacles
            if sqrt((robot_pos[0] - obs[0]) ** 2 + (robot_pos[1] - obs[1]) ** 2)
            <= Constants.VISION_RADIUS
        ]

        # Highlight obstacles in robot's vision (yellow circles)
        for obs in visible_obstacles:
            pygame.draw.circle(
                screen,
                (255, 255, 0),
                (int(obs[0] * grid_scale), int(obs[1] * grid_scale)),
                grid_scale // 8,
            )

        # Recalculate path only if enough time has passed since the last replanning
        current_time = pygame.time.get_ticks()
        if current_time - last_replan_time > 1000:  # 2-second delay
            new_start = Node(robot_pos[0], robot_pos[1], robot_theta, 0)
            new_path, new_search_tree_edges = hybrid_a_star(
                new_start, goal, obstacles, bounds, robot_pos
            )
            if new_path:  # Only update path if a valid path is found
                path = new_path
                search_tree_edges = new_search_tree_edges
                path_index = 0
            last_replan_time = current_time

        # Draw the predicted path (blue line)
        if path:
            for i in range(len(path) - 1):
                pygame.draw.line(
                    screen,
                    (0, 0, 255),
                    (int(path[i][0] * grid_scale), int(path[i][1] * grid_scale)),
                    (
                        int(path[i + 1][0] * grid_scale),
                        int(path[i + 1][1] * grid_scale),
                    ),
                    2,
                )

        # Move the robot along the path
        if path and path_index < len(path):
            target_x, target_y, target_theta = path[path_index]
            delta_x = target_x - robot_pos[0]
            delta_y = target_y - robot_pos[1]
            delta_theta = (
                target_theta - robot_theta + 180
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
                (vehicle_width_px, vehicle_height_px), pygame.SRCALPHA
            )
            pygame.draw.rect(
                vehicle_surface_prev,
                (0, 128, 255),
                (0, 0, vehicle_width_px, vehicle_height_px),
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
            (vehicle_width_px, vehicle_height_px), pygame.SRCALPHA
        )
        pygame.draw.rect(
            vehicle_surface,
            (0, 128, 255),
            (0, 0, vehicle_width_px, vehicle_height_px),
        )

        # Rotate the vehicle surface according to the robot's orientation
        rotated_vehicle = pygame.transform.rotate(vehicle_surface, -robot_theta)
        rotated_rect = rotated_vehicle.get_rect(
            center=(robot_pos[0] * grid_scale, robot_pos[1] * grid_scale)
        )

        # Blit the rotated vehicle onto the screen
        screen.blit(rotated_vehicle, rotated_rect.topleft)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


# Example usage
start = Node(0, 0, 0, 0)
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
