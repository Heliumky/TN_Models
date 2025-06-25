import os
import subprocess
from pathlib import Path
import time

# Get the current working directory
base_dir = Path.cwd()

# Get all subdirectories that start with 'R' and sort them
folders = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('R')])

max_parallel = 4  # Maximum number of jobs to run in parallel

# Record the starting time
start = time.time()

# Split the list of folders into batches of size 'max_parallel'
for i in range(0, len(folders), max_parallel):
    batch = folders[i:i+max_parallel]
    processes = []
    print(f"\n>>> Starting batch {i//max_parallel+1}: {[f.name for f in batch]}")

    # Launch each job in this batch
    for d in batch:
        print(f"Executing two_particles.py in {d.name} ...")
        p = subprocess.Popen(
            ['python', 'two_particles.py'],   # Run the script
            cwd=str(d),                        # Set working directory to the folder
            stdout=open(d/'nohup.out', 'w'),   # Save output to nohup.out
            stderr=subprocess.STDOUT           # Redirect errors to stdout
        )
        processes.append(p)

    # Wait for all jobs in this batch to finish
    for p in processes:
        p.wait()

    print(f"Batch {i//max_parallel+1} completed")

# Record the end time
end = time.time()
print(f"Total runtime: {(end-start)/3600:.4f} hr")
print("\nAll jobs completed")

