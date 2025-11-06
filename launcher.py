import subprocess
import sys
import time
import os
import re

# --- Parallel Training Launcher & Monitor ---

YEARS_TO_PROCESS = range(2015, 2025) # Processes 2015 through 2024

PROJECT_DIR = "/Users/ashishmishra/geopolitical_predictor"
PYTHON_EXECUTABLE = "/Users/ashishmishra/geopolitical_predictor/geo_venv/bin/python3"
MAPPER_SCRIPT = f"{PROJECT_DIR}/mapper.py"

def get_progress_from_log(log_path):
    """Reads the last line of a log file and parses tqdm progress."""
    try:
        with open(log_path, 'rb') as f:
            # Go to the end of the file
            f.seek(-2, os.SEEK_END)
            # Go backwards until a newline is found
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
            last_line = f.readline().decode()
        
        # Regex to find the percentage from a tqdm progress bar line
        match = re.search(r'(\d+)%\|', last_line)
        if match:
            return int(match.group(1))
        else:
            # If no match, progress is 0 or the line is not a progress bar
            return 0
    except (IOError, ValueError):
        # File might not exist yet or is empty
        return 0
    return 0

def run_launcher_and_monitor():
    """Launches mappers and then monitors their progress in a unified dashboard."""
    print("--- Starting Parallel Training Launcher ---")
    
    processes = {}
    log_paths = {}

    for year in YEARS_TO_PROCESS:
        log_path = f"{PROJECT_DIR}/{year}.log"
        log_paths[year] = log_path
        command = [PYTHON_EXECUTABLE, "-u", MAPPER_SCRIPT, str(year)]
        
        print(f"Launching mapper for {year}... Log: {log_path}")
        with open(log_path, 'w') as log_file:
            process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
            processes[year] = process

    print("\n--- All mappers launched. Starting monitoring dashboard... ---")
    time.sleep(2) # Give processes a moment to start

    try:
        all_done = False
        while not all_done:
            all_done = True
            total_progress = 0
            progress_bars = []

            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')
            print("--- Parallel Training Progress ---")

            for year in YEARS_TO_PROCESS:
                progress = get_progress_from_log(log_paths[year])
                total_progress += progress
                
                # Check if this specific process is still running
                exit_code = processes[year].poll()
                if exit_code is None:
                    all_done = False # At least one is still running
                    status = "Running"
                elif exit_code == 0:
                    # Process finished successfully
                    progress = 100
                    status = "Done   "
                else:
                    # Process terminated with an error
                    progress = get_progress_from_log(log_paths[year]) # Show last known progress
                    status = f"ERROR (Code: {exit_code})"

                # Create a simple text-based progress bar
                bar_length = 20
                filled_length = int(bar_length * progress // 100)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                progress_bars.append(f"  Year {year}: |{bar}| {progress:>3}% ({status})")

            # Print all progress bars
            print("\n".join(progress_bars))
            
            # Print overall progress
            overall_percentage = total_progress / len(YEARS_TO_PROCESS)
            print(f"\nOverall Progress: {overall_percentage:.2f}%")
            
            if all_done:
                break

            time.sleep(2) # Refresh rate of the dashboard

    except KeyboardInterrupt:
        print("\n--- Interrupted by user. Terminating mapper processes... ---")
        for year, process in processes.items():
            if process.poll() is None:
                print(f"  Terminating mapper for {year}...")
                process.terminate()
        print("All running mapper processes have been sent a termination signal.")
        sys.exit(1)

    print("\n--- All Mapper Processes Finished! ---")
    print("You can now run the reducer to combine the results:")
    print("  python3 /Users/ashishmishra/geopolitical_predictor/reducer.py")

if __name__ == "__main__":
    run_launcher_and_monitor()