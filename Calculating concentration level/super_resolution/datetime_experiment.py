# Time value in seconds
time_seconds = 3600

# Convert seconds to days, hours, minutes, and seconds
days = time_seconds // 86400
time_seconds %= 86400
hours = time_seconds // 3600
time_seconds %= 3600
minutes = time_seconds // 60
seconds = time_seconds % 60

# Format the time as DD:HH:MM:SS
formatted_time = "{:02d}:{:02d}:{:02d}:{:02d}".format(days, hours, minutes, seconds)

# Print the formatted time
print("Formatted time:", formatted_time)