import numpy as np
import matplotlib.pyplot as plt

path0 = '/Users/floracharbonnier/GitHub/MARL_local_electricity/outputs/results'

path_save = ''
path_record = path0+f"/run2026/record"
last = np.load(path_record + "/last_repeat0.npy", allow_pickle=True).item()
colours = ['darkviolet', 'deepskyblue', 'forestgreen']

tot_heat_cons = np.mean(np.sum(last['tot_E_heat']['baseline'], axis=0))
tot_ev_cons = np.mean(
    np.sum(
        last['batch']['loads_car'], axis=0
    )
)/2
tot_flex_cons = np.mean(np.sum(last['totcons']['baseline'], axis=0))

max_heat_cons = np.mean(np.max(last['tot_E_heat']['env_r_c'],axis=0))
max_ev_cons = np.mean(np.max(np.array(last['store']['env_r_c'][1:]) - np.array(last['store']['env_r_c'][:-1]),axis=0))
max_flex_cons = np.mean(np.max(last['flex_cons']['baseline'],axis=0))

n_homes_loads = 200
n_homes_ev = 140
n_homes_heat = 50

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
tot_cons = [tot_flex_cons, tot_ev_cons, tot_heat_cons]
n_homes = [n_homes_loads, n_homes_ev, n_homes_heat]
max_cons = [max_flex_cons, max_ev_cons, max_heat_cons]
labels = [
    'Flexible household loads',  'EV', 'Heat pump'
]


for tot_cons_i, n_homes_i, colour,label in zip(tot_cons, n_homes, colours, labels):
    axs[0].plot(tot_cons_i, n_homes_i, 'o', color=colour, label=label)
# axs[0].plot(
#     [tot_flex_cons, tot_ev_cons, tot_heat_cons],
#     [n_homes_loads, n_homes_ev, n_homes_heat],
#     'o',
#     color='gray'
# )
axs[0].legend()
axs[0].set_xlabel('Average consumption per day [kWh]')
axs[0].set_ylabel('Number of homes causing voltage constraint violations')
# axs[0].text(tot_heat_cons - 3, n_homes_heat + 10, 'Heat pump')
# axs[0].text(tot_ev_cons, n_homes_ev + 10, 'EV')
# axs[0].text(tot_flex_cons, n_homes_loads - 10, 'Flexible household loads')

for max_cons_i, n_homes_i, colour in zip(max_cons, n_homes, colours):
    axs[1].plot(max_cons_i, n_homes_i, 'o', color=colour)
# axs[1].plot(
#     [max_flex_cons, max_ev_cons, max_heat_cons],
#     [n_homes_loads, n_homes_ev, n_homes_heat],
#     'o',
#     color='gray'
# )
axs[1].set_xlabel('Maximum consumption per day [kWh]')
# axs[1].text(max_heat_cons, n_homes_heat + 10, 'Heat pump')
# axs[1].text(max_ev_cons, n_homes_ev + 10, 'EV')
# axs[1].text(max_flex_cons, n_homes_loads - 10, 'Flexible household loads')

fig.savefig("/Users/floracharbonnier/n_homes_violation_vs_loads_der.pdf", bbox_inches='tight', format='pdf', dpi=1200)
plt.close('all')


max_loads = np.max([tot_flex_cons, tot_ev_cons, tot_heat_cons])
n_homes_normalised_by_tot_loads = [
    n_homes_loads/tot_flex_cons, n_homes_ev/tot_ev_cons, n_homes_heat/tot_heat_cons
]
n_homes_normalised_by_max_loads = [
    n_homes_loads / max_flex_cons, n_homes_ev / max_ev_cons, n_homes_heat / max_heat_cons
]


# groups (4) = x
# girs/boys <> heat, ev, flex
X = ['normalised by total daily load',
     'normalised by maximum daily load']
Y_heat = [n_homes_heat/tot_heat_cons, n_homes_heat / max_heat_cons]
Y_ev = [n_homes_ev/tot_ev_cons, n_homes_ev / max_ev_cons]
Y_flex = [n_homes_loads/tot_flex_cons, n_homes_loads / max_flex_cons]

X_axis = np.arange(len(X))



import seaborn as sns
sns.color_palette("Set2")
fig = plt.figure()
plt.bar(X_axis - 0.2, Y_heat, 0.2, label='Heat pump'
        , color=colours[0])
plt.bar(X_axis, Y_ev, 0.2, label='EV'
        , color=colours[1])

plt.bar(X_axis + 0.2, Y_flex, 0.2, label='Flexible household loads'
        , color=colours[2])

plt.xticks(X_axis, X)
# plt.xlabel("Groups")
plt.ylabel("Number of homes causing voltage constraint violations\nnormalised by...")
# plt.title("Normalised number of homes causing voltage constraint violations")
plt.legend()
fig.gca().set_yscale('log')
plt.show()
fig.savefig("/Users/floracharbonnier/n_homes_violation_normalised.pdf", bbox_inches='tight', format='pdf', dpi=1200)
plt.close('all')
