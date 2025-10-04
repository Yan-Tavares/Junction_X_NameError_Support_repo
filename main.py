from eval.analyze import analyze_file

# Process any audio sample and get timestamps
timestamps = analyze_file("data/recordings/test.amr")
print(timestamps)
print("Done.")
# Returns: [5.23, 12.45, 23.67, ...]