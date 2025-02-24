import subprocess
import json
import os
import time
import shutil

# File paths
log_file_directory = "fixed_alpha_exp_ffi/"  # Standard directory where the log will be saved
default_log_file = "experiment_log.json"  # Default log file name

# Settings for weights
settings = {
    0: {"class_balance": 0.0, "num_examples": 1.0, "num_classes": 0.0, "avg_mi": 0.0, "feature_accuracy": 0.0, "label_distribution_balance": 0.0, "avg_ffi" : 0.0,},
    1: {"class_balance": 1.0, "num_examples": 0.0, "num_classes": 0.0, "avg_mi": 0.0, "feature_accuracy": 0.0, "label_distribution_balance": 0.0, "avg_ffi" : 0.0,},
    2: {"class_balance": 0.0, "num_examples": 0.0, "num_classes": 1.0, "avg_mi": 0.0, "feature_accuracy": 0.0, "label_distribution_balance": 0.0, "avg_ffi" : 0.0,},
    3: {"class_balance": 0.0, "num_examples": 0.0, "num_classes": 0.0, "avg_mi": 1.0, "feature_accuracy": 0.0, "label_distribution_balance": 0.0, "avg_ffi" : 0.0,},
    4: {"class_balance": 0.0, "num_examples": 0.0, "num_classes": 0.0, "avg_mi": 0.0, "feature_accuracy": 1.0, "label_distribution_balance": 0.0, "avg_ffi" : 0.0,},
    5: {"class_balance": 0.0, "num_examples": 0.0, "num_classes": 0.0, "avg_mi": 0.0, "feature_accuracy": 0.0, "label_distribution_balance": 1.0, "avg_ffi" : 0.0,},
    6: {"class_balance": 0.0, "num_examples": 0.0, "num_classes": 0.0, "avg_mi": 0.5, "feature_accuracy": 0.0, "label_distribution_balance": 0.5, "avg_ffi" : 0.0,},
    7: {"class_balance": 0.1, "num_examples": 0.0, "num_classes": 0.1, "avg_mi": 0.4, "feature_accuracy": 0.0, "label_distribution_balance": 0.4, "avg_ffi" : 0.0,},
    8: {"class_balance": 0.0, "num_examples": 0.0, "num_classes": 0.0, "avg_mi": 0.0, "feature_accuracy": 0.0, "label_distribution_balance": 0.0, "avg_ffi" : 1.0,},
}


settings_selection = [8] # Select settings
n_runs = 1 # Adjust the number of runs as needed

# Function to run the experiment with varying alpha
def run_experiment():
    for setting, weights in settings.items():

        if setting in settings_selection:

            print(f"Running with setting {setting}: {weights}")

            for run in range(1, n_runs + 1):
                print(f"Starting run {run} for setting {setting}...")

                try:
                    # Extract weights
                    class_balance = str(weights["class_balance"])
                    num_examples = str(weights["num_examples"])
                    num_classes = str(weights["num_classes"])
                    avg_mi = str(weights["avg_mi"])
                    feature_accuracy = str(weights["feature_accuracy"])
                    label_distribution_balance = str(weights["label_distribution_balance"])
                    avg_ffi = str(weights["avg_ffi"])

                    # Combine all key-value pairs into a single string
                    run_config = f"class_balance={class_balance} num_examples={num_examples} num_classes={num_classes} avg_mi={avg_mi} feature_accuracy={feature_accuracy} label_distribution_balance={label_distribution_balance} avg_ffi={avg_ffi}"

                    # Construct the command
                    command = ["flwr", "run", "--run-config", f"{run_config}"  # Enclose the run-config string in single quotes
                    ]

                    # Run the subprocess
                    result = subprocess.run(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )

                    # Debug the output
                    print("Command Output:", result.stdout)
                    print("Command Errors:", result.stderr)

                    shutil.move(default_log_file, f"{log_file_directory}setting_{setting}_run_{run}.json")

                except KeyError as e:
                    print(f"Missing required key in weights: {e}")
                except Exception as e:
                    print(f"An error occurred: {e}")


run_experiment()
