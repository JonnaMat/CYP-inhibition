"""Build the whole dataset for each task."""

# pylint: disable=no-name-in-module, import-error
from utils.data_preparation import data_preprocessing

# WARNING: Running this script for the first time takes a while.
tasks = ["CYP2C19", "CYP2D6"]#, "CYP3A4", "CYP1A2", "CYP2C9"]
for task in tasks:
    print(f"----- Starting with {task} -----")
    data = data_preprocessing(task)
