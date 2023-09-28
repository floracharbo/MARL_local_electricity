import os

root = 'outputs/results'
for run_folder in os.listdir(root):
    if run_folder[0] == '.':
        continue
    figures_path = os.path.join(root, run_folder, 'figures')
    for figure in os.listdir(figures_path):
        if not figure.endswith('.npy'):
            os.remove(os.path.join(figures_path, figure))
