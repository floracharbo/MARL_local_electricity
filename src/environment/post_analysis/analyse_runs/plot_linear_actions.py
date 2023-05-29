import matplotlib.pyplot as plt

green = (117 / 255, 189 / 255, 167 / 255)
grey = '#999999'
blue = '#377eb8'  # colorblind blue

fig = plt.figure(figsize=(6, 3))
plt.xlabel(r'$a_{battery}$ [-]')
plt.ylabel(r'$\Delta s$ [kWh]')

actions = [-1, 0, 1]

# for min(s) < 0 and max(s) < 0
mins = - 20
maxs = - 5
deltas = [mins, maxs, maxs]
plt.plot(actions, deltas, color=green, label=r'$\min(s) < 0$ and $\max(s) < 0$', lw=2)

# for min(s) < 0 and max(s) > 0
mins = - 20
maxs = 20
deltas = [mins, 0, maxs]
plt.plot(actions, deltas, color=grey, label=r'$\min(s) < 0$ and $\max(s) > 0$', lw=2)

# for min(s) > 0 and max(s) > 0
mins = 5
maxs = 20
deltas = [mins, mins, maxs]
plt.plot(actions, deltas, color=blue, label=r'$\min(s) > 0$ and $\max(s) > 0$', lw=2)

ax = plt.gca()
# ax.set_xticks([])
# ax.set_yticks([])
# plt.legend()
fig.savefig(
    "outputs/results_analysis/linear_battery_action.pdf",
    bbox_inches='tight', format='pdf', dpi=1200
)
