import os
import shutil
import subprocess

def sync_all_offline_runs(api_key, wandb_dir):
    """
    Syncs all offline W&B runs in the specified directory to the W&B server and moves synced runs to a 'sync_runs' directory.

    Parameters:
    - api_key (str): W&B API key for authentication.
    - wandb_dir (str): Path to the directory containing offline W&B runs.
    """
    if not os.path.exists(wandb_dir):
        print(f"The directory {wandb_dir} does not exist.")
        return

    # Directory for synced runs
    synced_dir = os.path.join(wandb_dir, "sync_runs")
    os.makedirs(synced_dir, exist_ok=True)

    # Find all offline run directories
    offline_runs = [d for d in os.listdir(wandb_dir) if d.startswith("offline-run-") or d.startswith("run-")]
    if not offline_runs:
        print("No offline runs found.")
        return

    print(f"Found {len(offline_runs)} offline run(s) to sync.")

    for run in offline_runs:
        run_path = os.path.join(wandb_dir, run)
        print(f"Syncing run: {run_path}")
        
        # Build the sync command
        sync_command = f"WANDB_API_KEY={api_key} wandb sync {run_path}"
        
        # Execute the sync command
        process = subprocess.Popen(
            sync_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # stdout, stderr = process.communicate()
        
        try:  
            while True:
                # Read line from stdout
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                    if "ERROR" in output:
                        print(f"Run {run} sync encountered an ERROR and was aborted.")
                        process.terminate()
                        break

                # Read line from stderr
                error = process.stderr.readline()
                if error == '' and process.poll() is not None:
                    break
                if error:
                    print(error.strip())
                    if "ERROR" in error:
                        print(f"Run {run} sync encountered an ERROR and was aborted.")
                        process.terminate()
                        break

            process.wait(timeout=10)

            # Output results
            # print("STDOUT:", stdout)
            # print("STDERR:", stderr)

            if process.returncode == 0:
                print(f"Run {run} synced successfully.")

                # Move the synced run to the synced_dir
                new_path = os.path.join(synced_dir, run)
                shutil.move(run_path, new_path)
                print(f"Moved {run} to {synced_dir}.")
            else:
                print(f"Failed to sync run {run}.")
        
        except subprocess.TimeoutExpired:
            print(f"Run {run_id} sync timed out.")
        except Exception as e:
            print(f"An error occurred while syncing run {run}: {e}")

if __name__ == "__main__":
    # Your W&B API key
    wandb_api_key = "c707377256b2e57dbb0b42bd3c36744b3d5617c8"
    
    # Path to the directory containing offline W&B runs
    wandb_directory = os.path.expanduser("./wandb/")

    sync_all_offline_runs(wandb_api_key, wandb_directory)

