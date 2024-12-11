from utils import *
import yaml

# Load configuration from YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
if __name__ == "__main__":
    # Initialize start, goal, obstacles, bounds
    start = Node(0, 0, 0)
    goal = Node(0, 0, 0)

    grid_size = config["grid_size"]
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

    # # Remove duplicates and sort
    # cross_lines_thick = sorted(set(cross_lines_thick))

    # all_points = set((x, y) for x in range(grid_size + 1) for y in range(grid_size + 1))

    # # Convert cross_lines to a set
    # cross_line_points = set(cross_lines_thick)

    # # Compute R as all grid points excluding cross_lines
    # R_points = all_points - cross_line_points

    # # Convert R_points back to a sorted list of tuples
    # R_points = sorted(R_points)
    inverted_grid = np.load("Town05_road_grid_origin.npy")
    # Determine obstacles from the grid
    # Assuming `inverted_grid` is a binary grid where 1 represents free space and 0 represents obstacles
    import code

    # code.interact(local=locals())
    obstacles = [
        (x - 50, y - 50)
        for x in range(inverted_grid.shape[0])
        for y in range(inverted_grid.shape[1])
        if inverted_grid[x, y] == 1
    ]
    # Remove duplicates and sort
    # obstacles = inverted_grid
    with open("Town05_road_grid_origin.txt", "r") as file:
        # Read lines and parse coordinates
        obstacles = []
        for line in file:
            try:
                x, y = map(float, line.strip().split(","))
                obstacles.append((x, y))
            except ValueError:
                continue  # Skip invalid lines

    # bounds = (0, 50, 0, 50)

    # Run simulation
    # planner = HybridAStar(start, goal, obstacles, bounds)

    simulation = Simulation(start, goal, obstacles)
    simulation.run()
