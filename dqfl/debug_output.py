import subprocess

# Run the command and capture output
result = subprocess.run(
    ["flwr", "run", "."],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Print both stdout and stderr for inspection
print("\n=== STDERR ===")
print(result.stderr)