import pygame
import numpy as np
import heapq
import threading
from threading import Lock
from math import sin, cos, tan, radians, degrees, sqrt, pi
import yaml
from hybridastar import HybridAStar, Node

# Load configuration from YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


# Pure Pursuit Control
def pure_pursuit_control(
    x, y, theta, waypoints, lookahead_distance, k_linear: float, k_angular: float
) -> float:
    max_dist = -1
    target_point = None
    # for idx, waypoint in enumerate(waypoints):
    #     pygame.draw.circle(screen, RED, transform_coords(waypoint[0], waypoint[1]), 5)
    for id, point in enumerate(reversed(waypoints)):
        dist = np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2)
        if dist <= lookahead_distance and dist > max_dist:
            max_dist = dist
            target_point = point
            break
    if target_point is None:  # No valid target
        return 0, 0
    # mark current target node
    # pygame.draw.circle(
    #     screen, GREEN, transform_coords(target_point[0], target_point[1]), 8
    # )
    dx, dy = target_point[0] - x, target_point[1] - y
    target_theta = np.arctan2(dy, dx)
    # error
    heading_err = target_theta - theta
    dist_err = np.sqrt(dx**2 + dy**2)
    omega = k_angular * heading_err
    v = k_linear * dist_err
    # waiting for rotation if high heading error
    if abs(heading_err) >= (45.0 / 180.0) * np.pi:
        v = 0.0
    return v, omega


class Constants:
    RESOLUTION = config["resolution"]
    ANGLE_RESOLUTION = config["angle_resolution"]
    GOAL_TOLERANCE = config["goal_tolerance"]
    SCREEN_SIZE = config["screen_size"]
    GRID_SIZE = config["grid_size"]
    VISION_RADIUS = config["vision_radius"]
    VEHICLE_WIDTH = config["vehicle_width"]
    VEHICLE_LENGTH = config["vehicle_length"]
    WHEEL_BASE = config["wheel_base"]
    MAX_STEERING_ANGLE = config["max_steering_angle"]
    STEP_SIZE = config["step_size"]
    OFFSET_X = config["offset_x"]
    OFFSET_Y = config["offset_y"]

    GRID_SCALE = SCREEN_SIZE // GRID_SIZE
    TURNING_RADIUS = WHEEL_BASE / sin(radians(MAX_STEERING_ANGLE))


class Simulation:
    def __init__(self, start, goal, obstacles):
        # Initialize variables, pygame, planner, etc.

        pygame.init()
        pygame.key.set_repeat(200, 50)
        self.screen = pygame.display.set_mode(
            (Constants.SCREEN_SIZE, Constants.SCREEN_SIZE)
        )
        self.clock = pygame.time.Clock()
        self.running = True
        self.angle_resolution = Constants.ANGLE_RESOLUTION
        self.planner = HybridAStar(
            obstacles=obstacles,
            angle_resolution=self.angle_resolution,
            resolution=Constants.RESOLUTION,
        )

        self.start = start
        self.goal = goal
        self.obstacles = obstacles

        self.grid_scale = Constants.GRID_SCALE
        self.robot_pos = [start.x, start.y]
        self.robot_theta = start.theta

        self.found = self.planner.search(self.start, self.goal)
        if not self.found:
            print("No path found!")
        self.path = self.planner.path
        self.path_index = 0
        self.last_replan_time = 0

        # Define the robot size in pixels
        self.vehicle_width_px = Constants.VEHICLE_WIDTH * self.grid_scale
        self.vehicle_length_px = Constants.VEHICLE_LENGTH * self.grid_scale

        self.previous_positions = []
        self.last_recorded_pos = [
            self.robot_pos[0],
            self.robot_pos[1],
            self.robot_theta,
        ]

        self.offset_x = Constants.OFFSET_X
        self.offset_y = Constants.OFFSET_Y

        self.planner_lock = Lock()
        self.planning = False
        self.planner_thread = None
        self.waypoint = None

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Handle mouse click events
                self.handle_mouse_click(event)

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                self.planning = False
                self.planning = self.handle_set_goal()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.offset_x += self.grid_scale
                elif event.key == pygame.K_RIGHT:
                    self.offset_x -= self.grid_scale
                elif event.key == pygame.K_UP:
                    self.offset_y += self.grid_scale
                elif event.key == pygame.K_DOWN:
                    self.offset_y -= self.grid_scale

    def handle_mouse_click(self, event):
        # Get mouse position
        mouse_x, mouse_y = pygame.mouse.get_pos()

        # Convert mouse position to grid coordinates
        grid_x = (mouse_x - self.offset_x) // self.grid_scale
        grid_y = (mouse_y - self.offset_y) // self.grid_scale
        print(grid_x, grid_y)

        # Add obstacle to the list if not already present
        if (grid_x, grid_y) not in self.obstacles:
            self.obstacles.append((grid_x, grid_y))
            # Replan
            self.update_planner(replan=True)

    def handle_set_goal(self):
        # Set new goal
        mouse_x, mouse_y = pygame.mouse.get_pos()
        grid_x = (mouse_x - self.offset_x) / self.grid_scale
        grid_y = (mouse_y - self.offset_y) / self.grid_scale
        self.goal = Node(grid_x, grid_y, 90, 0)
        # Replan
        self.update_planner(replan=True)

    def run_planner(self):
        # with self.planner_lock:
        #     self.planner.goal = self.goal

        found = False
        # Now run the search outside the lock if the search uses open_set frequently
        # If needed, lock/unlock around heapq operations inside the search method as well.
        found = self.planner.search(
            Node(self.robot_pos[0], self.robot_pos[1], 0, 0), self.goal
        )  # Make sure search does not modify open_set without a lock
        # Update results back under lock if modifying shared data
        with self.planner_lock:
            self.found = found
            if not found:
                print("No path found!")
            self.path = self.planner.path
            self.path_index = 0
            self.planning = False

            self.waypoint = [(p.x, p.y, p.theta) for p in self.path]
            print(self.waypoint)

    def update_simulation(self):
        # lookahead_distance = 5
        # k_linear = 5.0
        # k_angular = 5.0
        # dt = 0.01  # Time step
        # if self.waypoint is not None:
        #     v, omega = pure_pursuit_control(
        #         self.robot_pos[0],
        #         self.robot_pos[1],
        #         self.robot_theta,
        #         self.waypoint,
        #         lookahead_distance,
        #         k_linear,
        #         k_angular,
        #     )
        # else:
        #     v = 0
        #     omega = 0
        # self.robot_pos[0] += v * np.cos(self.robot_theta) * dt
        # self.robot_pos[1] += v * np.sin(self.robot_theta) * dt
        # self.robot_theta += omega * dt

        # Move the robot along the current path if available
        # If still planning, we keep using the old path stored in self.path

        if self.path and self.path_index < len(self.path):
            target_node = self.path[self.path_index]
            delta_x = target_node.x - self.robot_pos[0]
            delta_y = target_node.y - self.robot_pos[1]

            delta_theta = (
                target_node.theta - self.robot_theta + 180 + 360
            ) % 360 - 180  # Shortest angle

            self.robot_pos[0] += delta_x * 1
            self.robot_pos[1] += delta_y * 1
            self.robot_theta += delta_theta * 1

            if sqrt(delta_x**2 + delta_y**2) < 0.1 and abs(delta_theta) < 1:
                self.path_index += 1

        # Record previous position if moved enough
        if (
            sqrt(
                (self.robot_pos[0] - self.last_recorded_pos[0]) ** 2
                + (self.robot_pos[1] - self.last_recorded_pos[1]) ** 2
            )
            >= 0.5
        ):
            self.previous_positions.append(
                (
                    self.last_recorded_pos[0],
                    self.last_recorded_pos[1],
                    self.robot_theta,
                )
            )
        self.last_recorded_pos = [
            self.robot_pos[0],
            self.robot_pos[1],
            self.robot_theta,
        ]

    def update_planner(self, replan=False):
        # Only start re-planning if not already planning
        if replan and not self.planning:
            self.planning = True
            # Start the planner thread
            self.planner_thread = threading.Thread(target=self.run_planner)
            self.planner_thread.start()

    def draw(self):
        self.screen.fill((255, 255, 255))

        # Draw grid with offset
        for x in range(0, Constants.SCREEN_SIZE, self.grid_scale):
            pygame.draw.line(
                self.screen,
                (200, 200, 200),
                (x - self.grid_scale // 2, 0),
                (
                    x - self.grid_scale // 2,
                    Constants.SCREEN_SIZE - self.grid_scale // 2,
                ),
            )
        for y in range(0, Constants.SCREEN_SIZE, self.grid_scale):
            pygame.draw.line(
                self.screen,
                (200, 200, 200),
                (0, y - self.grid_scale // 2),
                (
                    Constants.SCREEN_SIZE - self.grid_scale // 2,
                    y - self.grid_scale // 2,
                ),
            )

        # Draw obstacles with offset
        for obs in self.obstacles:
            pygame.draw.rect(
                self.screen,
                (20, 0, 20),  # color for obstacles
                (
                    obs[0] * self.grid_scale
                    - self.grid_scale // 2
                    + self.offset_x,  # Top-left x position with offset
                    (obs[1] - 0.5) * self.grid_scale
                    + self.offset_y,  # Top-left y position with offset
                    self.grid_scale,  # Width of the grid cell
                    self.grid_scale,  # Height of the grid cell
                ),
            )

        # Draw start and goal with offset
        start_rect = pygame.Rect(
            int(
                self.start.x * self.grid_scale
                - self.vehicle_length_px / 2
                + self.offset_x
            ),
            int(
                self.start.y * self.grid_scale
                - self.vehicle_width_px / 2
                + self.offset_y
            ),
            self.vehicle_length_px,
            self.vehicle_width_px,
        )
        pygame.draw.rect(self.screen, (0, 255, 0), start_rect, 1)

        goal_center = (
            int(
                self.goal.x * self.grid_scale + self.offset_x
            ),  # X-coordinate with offset
            int(
                self.goal.y * self.grid_scale + self.offset_y
            ),  # Y-coordinate with offset
        )

        goal_radius = int(Constants.GOAL_TOLERANCE * self.grid_scale)

        # Draw the goal circle with offset
        pygame.draw.circle(
            self.screen, (255, 0, 255), goal_center, goal_radius, 1
        )  # 1 is the thickness

        # Draw the search tree (yellow lines) with offset
        for edge in self.planner.search_tree_edges:
            parent, child = edge
            pygame.draw.line(
                self.screen,
                (255, 200, 0),
                (
                    int(parent.x * self.grid_scale) + self.offset_x,
                    int(parent.y * self.grid_scale) + self.offset_y,
                ),
                (
                    int(child.x * self.grid_scale) + self.offset_x,
                    int(child.y * self.grid_scale) + self.offset_y,
                ),
                1,
            )

        # Draw robot vision with offset
        pygame.draw.circle(
            self.screen,
            (0, 255, 255),
            (
                int(self.robot_pos[0] * self.grid_scale) + self.offset_x,
                int(self.robot_pos[1] * self.grid_scale) + self.offset_y,
            ),
            Constants.VISION_RADIUS * self.grid_scale,
            1,
        )

        # Draw the previous positions as thin-lined rotated rectangles with offset
        for pos_x, pos_y, pos_theta in self.previous_positions:
            # Create a surface for the vehicle
            vehicle_surface_prev = pygame.Surface(
                (self.vehicle_length_px, self.vehicle_width_px), pygame.SRCALPHA
            )
            pygame.draw.rect(
                vehicle_surface_prev,
                (0, 128, 255),
                (0, 0, self.vehicle_length_px, self.vehicle_width_px),
                1,  # Outline only
            )
            # Rotate the surface
            rotated_vehicle_prev = pygame.transform.rotate(
                vehicle_surface_prev, -pos_theta
            )
            rotated_rect_prev = rotated_vehicle_prev.get_rect(
                center=(
                    pos_x * self.grid_scale + self.offset_x,
                    pos_y * self.grid_scale + self.offset_y,
                )
            )
            self.screen.blit(rotated_vehicle_prev, rotated_rect_prev.topleft)

        # Draw the robot as a rotated rectangle with offset
        vehicle_surface = pygame.Surface(
            (self.vehicle_length_px, self.vehicle_width_px), pygame.SRCALPHA
        )
        pygame.draw.rect(
            vehicle_surface,
            (0, 128, 255),
            (0, 0, self.vehicle_length_px, self.vehicle_width_px),
        )

        # Rotate the vehicle surface according to the robot's orientation
        rotated_vehicle = pygame.transform.rotate(vehicle_surface, -self.robot_theta)
        rotated_rect = rotated_vehicle.get_rect(
            center=(
                self.robot_pos[0] * self.grid_scale + self.offset_x,
                self.robot_pos[1] * self.grid_scale + self.offset_y,
            )
        )

        # Blit the rotated vehicle onto the screen
        self.screen.blit(rotated_vehicle, rotated_rect.topleft)

        # Draw the predicted path (blue line) with offset
        if self.path:
            for i in range(len(self.path) - 1):
                pygame.draw.line(
                    self.screen,
                    (0, 0, 255),
                    (
                        int(self.path[i].x * self.grid_scale) + self.offset_x,
                        int(self.path[i].y * self.grid_scale) + self.offset_y,
                    ),
                    (
                        int(self.path[i + 1].x * self.grid_scale) + self.offset_x,
                        int(self.path[i + 1].y * self.grid_scale) + self.offset_y,
                    ),
                    2,
                )

        pygame.display.flip()
        self.clock.tick(60)

    def run(self):
        while self.running:
            self.handle_events()
            self.update_simulation()
            self.draw()
        pygame.quit()
