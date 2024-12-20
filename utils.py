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
    GRID_SCALE = SCREEN_SIZE // GRID_SIZE
    TURNING_RADIUS = WHEEL_BASE / sin(radians(MAX_STEERING_ANGLE))


class Simulation:
    def __init__(self, start, goal, obstacles, bounds):
        # Initialize variables, pygame, planner, etc.
        pygame.init()
        self.screen = pygame.display.set_mode(
            (Constants.SCREEN_SIZE, Constants.SCREEN_SIZE)
        )
        self.clock = pygame.time.Clock()
        self.running = True
        self.planner = HybridAStar(start, obstacles, bounds)
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.bounds = bounds

        self.grid_scale = Constants.GRID_SCALE
        self.robot_pos = [start.x, start.y]
        self.robot_theta = start.theta

        self.planner.goal = goal
        self.found = self.planner.search()
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

        self.planner_lock = Lock()
        self.planning = False
        self.planner_thread = None

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

    def handle_mouse_click(self, event):
        # Get mouse position
        mouse_x, mouse_y = pygame.mouse.get_pos()

        # Convert mouse position to grid coordinates
        grid_x = mouse_x // self.grid_scale
        grid_y = mouse_y // self.grid_scale

        # Add obstacle to the list if not already present
        if (grid_x, grid_y) not in self.obstacles:
            self.obstacles.append((grid_x, grid_y))
            # Replan
            self.update_planner(replan=True)

    def handle_set_goal(self):
        # Set new goal
        mouse_x, mouse_y = pygame.mouse.get_pos()
        grid_x = mouse_x / self.grid_scale
        grid_y = mouse_y / self.grid_scale
        self.goal = Node(grid_x, grid_y, 0, 0)
        # Replan
        self.update_planner(replan=True)

    def run_planner(self):
        with self.planner_lock:
            # Prepare planner state
            self.planner.start = Node(
                self.robot_pos[0], self.robot_pos[1], self.robot_theta, 0
            )
            self.planner.obstacles = self.obstacles
            self.planner.bounds = self.bounds
            self.planner.open_set = []
            self.planner.closed_set = set()
            self.planner.motion_primitives = self.planner.generate_motion_primitives()
            self.planner.obstacle_map = self.planner.create_obstacle_map()
            self.planner.path = []
            self.planner.search_tree_edges = []
            self.planner.goal = self.goal

        found = False
        # Now run the search outside the lock if the search uses open_set frequently
        # If needed, lock/unlock around heapq operations inside the search method as well.
        found = (
            self.planner.search()
        )  # Make sure search does not modify open_set without a lock

        # Update results back under lock if modifying shared data
        with self.planner_lock:
            self.found = found
            if not found:
                print("No path found!")
            self.path = self.planner.path
            self.path_index = 0
            self.planning = False

    def update_simulation(self):
        pass
        # Move the robot along the path
        # if self.path and self.path_index < len(self.path):
        #     target_node = self.path[self.path_index]
        #     delta_x = target_node.x - self.robot_pos[0]
        #     delta_y = target_node.y - self.robot_pos[1]
        #     delta_theta = (
        #         target_node.theta - self.robot_theta + 180 + 360
        #     ) % 360 - 180  # Shortest angle

        #     self.robot_pos[0] += delta_x * 1
        #     self.robot_pos[1] += delta_y * 1
        #     self.robot_theta += delta_theta * 1

        #     if sqrt(delta_x**2 + delta_y**2) < 0.1 and abs(delta_theta) < 1:
        #         self.path_index += 1

        #     # Record previous position if moved enough
        #     if (
        #         sqrt(
        #             (self.robot_pos[0] - self.last_recorded_pos[0]) ** 2
        #             + (self.robot_pos[1] - self.last_recorded_pos[1]) ** 2
        #         )
        #         >= 0.5
        #     ):
        #         self.previous_positions.append(
        #             (
        #                 self.last_recorded_pos[0],
        #                 self.last_recorded_pos[1],
        #                 self.robot_theta,
        #             )
        #         )
        #         self.last_recorded_pos = [
        #             self.robot_pos[0],
        #             self.robot_pos[1],
        #             self.robot_theta,
        #         ]

    def update_planner(self, replan=False):
        # Only start re-planning if not already planning
        if replan and not self.planning:
            self.planning = True
            # Start the planner thread
            self.planner_thread = threading.Thread(target=self.run_planner)
            self.planner_thread.start()

    def draw(self):
        self.screen.fill((255, 255, 255))

        # Draw grid
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

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(
                self.screen,
                (20, 0, 20),  # color for obstacles
                (
                    obs[0] * self.grid_scale
                    - self.grid_scale // 2,  # Top-left x position of the grid cell
                    (obs[1] - 0.5)
                    * self.grid_scale,  # Top-left y position of the grid cell
                    self.grid_scale,  # Width of the grid cell
                    self.grid_scale,  # Height of the grid cell
                ),
            )

        # Draw start and goal
        start_rect = pygame.Rect(
            int(self.start.x * self.grid_scale - self.vehicle_length_px / 2),
            int(self.start.y * self.grid_scale - self.vehicle_width_px / 2),
            self.vehicle_length_px,
            self.vehicle_width_px,
        )
        pygame.draw.rect(self.screen, (0, 255, 0), start_rect, 1)

        goal_center = (
            int(self.goal.x * self.grid_scale),  # X-coordinate of the goal's center
            int(self.goal.y * self.grid_scale),  # Y-coordinate of the goal's center
        )

        goal_radius = int(Constants.GOAL_TOLERANCE * self.grid_scale)

        # Draw the goal circle on the screen
        pygame.draw.circle(
            self.screen, (255, 0, 255), goal_center, goal_radius, 1
        )  # 1 is the thickness

        # Draw the search tree (yellow lines)
        for edge in self.planner.search_tree_edges:
            parent, child = edge
            pygame.draw.line(
                self.screen,
                (255, 200, 0),
                (int(parent.x * self.grid_scale), int(parent.y * self.grid_scale)),
                (int(child.x * self.grid_scale), int(child.y * self.grid_scale)),
                1,
            )

        # Draw robot vision
        pygame.draw.circle(
            self.screen,
            (0, 255, 255),
            (
                int(self.robot_pos[0] * self.grid_scale),
                int(self.robot_pos[1] * self.grid_scale),
            ),
            Constants.VISION_RADIUS * self.grid_scale,
            1,
        )

        # Draw the previous positions as thin-lined rotated rectangles
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
                center=(pos_x * self.grid_scale, pos_y * self.grid_scale)
            )
            self.screen.blit(rotated_vehicle_prev, rotated_rect_prev.topleft)

        # Draw the robot as a rotated rectangle
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
                self.robot_pos[0] * self.grid_scale,
                self.robot_pos[1] * self.grid_scale,
            )
        )

        # Blit the rotated vehicle onto the screen
        self.screen.blit(rotated_vehicle, rotated_rect.topleft)

        # Draw the predicted path (blue line)
        if self.path:
            for i in range(len(self.path) - 1):
                pygame.draw.line(
                    self.screen,
                    (0, 0, 255),
                    (
                        int(self.path[i].x * self.grid_scale),
                        int(self.path[i].y * self.grid_scale),
                    ),
                    (
                        int(self.path[i + 1].x * self.grid_scale),
                        int(self.path[i + 1].y * self.grid_scale),
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
