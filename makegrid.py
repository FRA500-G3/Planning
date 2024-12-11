import carla
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

import scipy.ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt


def fill_road_right_to_left_with_inversion(carla_map, grid_size=1.0):
    """
    Fill the road in a grid world with 1s from right to left in both x and y directions.
    Apply binary closing to remove holes, and invert values.

    :param carla_map: The CARLA map object.
    :param grid_size: Size of each grid cell in meters.
    """
    # Get topology
    topology = carla_map.get_topology()

    # Find map boundaries
    all_waypoints = [
        wp for segment in topology for wp in segment[0].next_until_lane_end(grid_size)
    ]
    min_x = min(wp.transform.location.x for wp in all_waypoints)
    max_x = max(wp.transform.location.x for wp in all_waypoints)
    min_y = min(wp.transform.location.y for wp in all_waypoints)
    max_y = max(wp.transform.location.y for wp in all_waypoints)

    # Define grid dimensions
    grid_width = int((max_x - min_x) / grid_size) + 1
    grid_height = int((max_y - min_y) / grid_size) + 1

    # Create the grid world
    grid = np.zeros((grid_height, grid_width), dtype=int)

    # Fill the grid from right to left for each road
    for segment in topology:
        start_wp = segment[0]
        waypoints = start_wp.next_until_lane_end(grid_size)

        for wp in waypoints:
            # Get the right border of the lane
            right_x = wp.transform.location.x - wp.lane_width / 2
            right_y = wp.transform.location.y

            # Fill grid cells progressively from right to left
            for offset_x in np.arange(0, wp.lane_width, grid_size):
                for offset_y in np.arange(0, wp.lane_width, grid_size):
                    x = right_x + offset_x
                    y = right_y + offset_y

                    # Convert world coordinates to grid indices
                    grid_x = int((x - min_x) / grid_size)
                    grid_y = int((y - min_y) / grid_size)

                    # Mark the cell as road
                    if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                        grid[grid_y, grid_x] = 1

    d = 5
    closed_grid = ndi.binary_closing(grid, structure=np.ones((d, d))).astype(int)

    # Invert the grid values
    inverted_grid = 1 - closed_grid

    # Plot the grid
    plt.figure(figsize=(12, 12))
    plt.imshow(
        inverted_grid, cmap="Greys", origin="lower", extent=[min_x, max_x, min_y, max_y]
    )
    plt.colorbar(label="Road Presence (1 = No Road, 0 = Road)")
    plt.title("Inverted Grid World with Roads Filled Right to Left")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    return inverted_grid


def get_map_size(client, waypoint_distance: float = 2.0):
    """
    Calculate the size of the CARLA map by determining its bounding box.

    Args:
        client (carla.Client): The CARLA client object connected to the simulator.
        waypoint_distance (float): Distance between waypoints for estimation. Default is 2.0 meters.

    Returns:
        dict: A dictionary containing map size and bounding box coordinates:
              {
                  "map_size_x": float,
                  "map_size_y": float,
                  "map_size_z": float,
                  "bounding_box": {
                      "min": (float, float, float),
                      "max": (float, float, float)
                  }
              }
    """
    # Access the world and map
    world = client.get_world()
    carla_map = world.get_map()

    # Generate waypoints
    waypoints = carla_map.generate_waypoints(waypoint_distance)

    # Initialize bounds
    min_x, min_y, min_z = float("inf"), float("inf"), float("inf")
    max_x, max_y, max_z = float("-inf"), float("-inf"), float("-inf")

    # Find the bounding box
    for waypoint in waypoints:
        location = waypoint.transform.location
        min_x = min(min_x, location.x)
        min_y = min(min_y, location.y)
        min_z = min(min_z, location.z)
        max_x = max(max_x, location.x)
        max_y = max(max_y, location.y)
        max_z = max(max_z, location.z)

    # Calculate map size
    map_size_x = max_x - min_x
    map_size_y = max_y - min_y
    map_size_z = max_z - min_z

    return {
        "map_size_x": map_size_x,
        "map_size_y": map_size_y,
        "map_size_z": map_size_z,
        "bounding_box": {"min": (min_x, min_y, min_z), "max": (max_x, max_y, max_z)},
    }


# Connect to CARLA
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
map_name = "Town07"
map_info = get_map_size(client)
print(
    f"{map_name} Map Size (X, Y): {map_info['map_size_x']:.2f} m, {map_info['map_size_y']:.2f} m"
)

# Get the world and map
world = client.get_world()
carla_map = world.get_map()


# Fill the road and visualize the grid world
grid = fill_road_right_to_left_with_inversion(carla_map)

# Save the grid as a NumPy file
file_name = f"{map_name}_road_grid.npy"
np.save(file_name, grid)
print(f"Grid saved as '{file_name}'.")
