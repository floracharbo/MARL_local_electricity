import os
import shutil

# DELETE_FOLDER = 'figures'
DELETE_FOLDER = 'data'
root = 'outputs/results'
for run_folder in os.listdir(root):
    if run_folder[0] == '.':
        continue
    figures_path = os.path.join(root, run_folder, DELETE_FOLDER)
    if DELETE_FOLDER == 'data' and os.path.exists(figures_path):
        shutil.rmtree(figures_path)
    elif DELETE_FOLDER == 'figures':
        for figure in os.listdir(figures_path):
            if not figure.endswith('.npy'):
                os.remove(os.path.join(figures_path, figure))
