def plot_time_performance_layers():
    times = np.array([
        52.53393412,
        56.30612707,
        69.72513795,
        317.7233688,
        1403.634051,
        62.27527117,
        51.00517082,
        76.54939389,
        408.2064509,
        1995.955998,
        45.3483171,
        93.336627,
        71.67132807,
        502.161931,
        2852.27796483039
    ]) / 60
    savings = np.array([
        0.403742986,
        0.428222236,
        0.429513,
        0.4621638,
        0.43779477,
        0.398818979,
        0.45392288,
        0.4571914,
        0.451994817,
        0.46654132,
        0.368527888,
        0.409844266,
        0.435698557,
        0.447063209,
        0.4695001815
    ]) * 24 * 365/12
    layer_sizes = [1e2, 5e2, 1e3, 5e3, 1e4]
    all_layer_sizes = layer_sizes * 3
    times_per_layer = [times[layer * 5: (layer + 1) * 5] for layer in range(3)]
    savings_per_layer = [savings[layer * 5: (layer + 1) * 5] for layer in range(3)]
    fig, axs = plt.subplots(2)
    colors_idxs = np.linspace(0, 1, 3)

    for layer, color_idx in enumerate(colors_idxs):
        axs[0].plot(layer_sizes, times_per_layer[layer], label=f"{layer + 1} hidden layers", color=plt.cm.cool(color_idx))
        axs[1].plot(layer_sizes, savings_per_layer[layer], label=f"{layer + 1} hidden layers", color=plt.cm.cool(color_idx))
    axs[0].legend()
    axs[1].legend()
    axs[0].set_ylabel("Time [mins]")
    axs[0].set_xlabel("Hidden layer dimension")
    axs[1].set_ylabel("Savings [£/home/month]")
    axs[1].set_xlabel("Hidden layer dimension")
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')

    axs[1].set_xscale('log')
    fig.savefig("outputs/results_analysis/Layers_dimensions_tradeoffs_suplots")
    plt.close("all")

    fig = plt.figure()
    for layer, color_idx in enumerate(colors_idxs):
        for time, saving, layer_size in zip(times_per_layer[layer], savings_per_layer[layer], layer_sizes):
            horizontalalignment = 'right' if time > 40 else 'left'
            plt.text(time, saving, f'{layer + 1} x' + f'{layer_size:.0E}', horizontalalignment=horizontalalignment)

    z = [time * 60 /saving for time, saving in zip(times, savings)]
    scat = plt.gca().scatter(times, savings, c=z, marker="o", cmap="jet")
    plt.colorbar(scat, label='[sec/(£/month/agent)]')
    plt.ylabel("Savings [£/home/month]")
    plt.xlabel("Time [mins]")
    plt.tight_layout()

    fig.savefig("outputs/results_analysis/Layers_dimensions_tradeoffs_2d")



def rename_runs():
    folders = os.listdir(results_path)
    i0 = 703
    initial_numbers = sorted([int(folder[3:]) for folder in folders if folder[0: 3] == "run"])[:-1]
    for i, initial_number in enumerate(initial_numbers):
        if initial_number != i0 + i:
            os.rename(results_path / f"run{initial_number}", results_path / f"run{i0 + i}")
            print(f"rename run{initial_number} -> run{i0 + i}")

    runs = list(range(730, 764))
    for i in range(len(runs)):
        run = runs[- (i+1)]
        os.rename(results_path / f"run{run}", results_path / f"run{run + 1}")
        print(f"rename run{run} -> {run + 1}")