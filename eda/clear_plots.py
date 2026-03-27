# Delete all generated plot images and their folders
import os
import shutil

plot_root = os.path.join(os.path.dirname(__file__), "plots")

if not os.path.isdir(plot_root):
    print("No plots folder found, nothing to clear.")
else:
    shutil.rmtree(plot_root)
    print("Cleared", plot_root)
