import math

# List of degrees
degrees_list = [0.1, -5.6, -0, 52.7, 0.4, 58.3, 0.5]

# Convert to radians
radians_list = [math.radians(deg) for deg in degrees_list]

# Print results
print("Degrees:", degrees_list)
print("Radians:", radians_list)