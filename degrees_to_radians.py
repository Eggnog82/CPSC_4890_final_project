import math

# List of degrees
degrees_list = [-0.3, -44.6, -0.6, 21.1, 0.4, 67.4, 0.7]

# Convert to radians
radians_list = [math.radians(deg) for deg in degrees_list]

# Print results
print("Degrees:", degrees_list)
print("Radians:", radians_list)