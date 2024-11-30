from utils import *
import yaml

if __name__ == "__main__":
    # Initialize start, goal, obstacles, bounds
    start = Node(8, 8, 0, 0)
    goal = Node(40, 40, 0, 0)

    grid_size = 50
    lane_thickness = 1
    lane_spacing = 8
    cross_lines_thick = []

    # Add horizontal and vertical lines with 2-block thickness
    for i in range(0, grid_size + 1, lane_spacing):
        # Horizontal lines with thickness
        for t in range(-lane_thickness // 2, lane_thickness // 2 + 1):
            cross_lines_thick.extend(
                [(x, i + t) for x in range(grid_size + 1) if 0 <= i + t <= grid_size]
            )
        # Vertical lines with thickness
        for t in range(-lane_thickness // 2, lane_thickness // 2 + 1):
            cross_lines_thick.extend(
                [(i + t, y) for y in range(grid_size + 1) if 0 <= i + t <= grid_size]
            )

    # Remove duplicates and sort
    cross_lines_thick = sorted(set(cross_lines_thick))

    all_points = set((x, y) for x in range(grid_size + 1) for y in range(grid_size + 1))

    # Convert cross_lines to a set
    cross_line_points = set(cross_lines_thick)

    # Compute R as all grid points excluding cross_lines
    R_points = all_points - cross_line_points

    # Convert R_points back to a sorted list of tuples
    R_points = sorted(R_points)

    # Remove duplicates and sort
    obstacles = R_points

    bounds = (0, 50, 0, 50)

    # Run simulation
    # planner = HybridAStar(start, goal, obstacles, bounds)

    simulation = Simulation(start, goal, obstacles, bounds)
    simulation.run()

    # self.start = start
    # self.obstacles = obstacles
    # self.bounds = bounds
    # self.open_set = []
    # self.closed_set = set()
    # self.motion_primitives = self.generate_motion_primitives()
    # self.path = []
    # self.search_tree_edges = []
    # self.grid_size_phi = int(360 / Constants.ANGLE_RESOLUTION)
    # self.map_width = int((bounds[1] - bounds[0]) / Constants.RESOLUTION)
    # self.map_height = int((bounds[3] - bounds[2]) / Constants.RESOLUTION)
    # self.obstacle_map = self.create_obstacle_map()
