import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from collections import Counter
from matplotlib import colors
from matplotlib.legend import Legend
from matplotlib.pyplot import viridis
from typing import (
    Tuple, 
    Dict, 
    List,
    Literal
)

from sympy.logic import true
from pandemic_control.environment import ACTIONS_STRUCT

# Keep consistency within colors
dynamics_cmap = {
    'Susceptible':'tab:blue',
    'Exposed':'tab:orange',
    'Infected':'tab:red',
    'Infected_cumul':'tab:red',
    'Infected_ref': 'xkcd:red',
    'Recovered':'tab:green',
    'Recovered_ref': 'xkcd:green',
    'Deceased':'tab:brown',
    'Deceased_ref':'xkcd:brown',
    'Symptomatic':'tab:pink',
    'Asymptomatic':'tab:cyan',
    'Hospitalized':'tab:grey',
    'Hospitalized_ref':'xkcd:blue',
    'Vaccinated':'tab:purple',
    'Economy':'tab:red',
    'Actions':'tab:green',
    'Health':'tab:orange',
    'Reward':'tab:blue',
    'Infected_rew': 'xkcd:red',
    'Hospitalized_rew': 'tab:grey',
    'Deceased_rew': 'xkcd:brown',
    'Deceased_cumul':'xkcd:brown',
    'Recovered_cumul':'tab:green', 
    'Vaccinated_cumul':'tab:purple',
}

comparison_cmap = {
    'Hospitalized':'tab:orange',
    'Hospitalized_ref':'xkcd:orange',
    'Recovered':'tab:green',
    'Recovered_ref': 'xkcd:green',
    'Deceased':'tab:grey',
    'Deceased_ref':'xkcd:grey',
    'Infected':'tab:red',
    'Infected_cum':'tab:red',
    'Infected_ref': 'xkcd:red',
}

# Labels per environment
env_labels_map = {
    'SIR': [
        'Susceptible',
        'Infected',
        'Recovered'
    ], 
    'SEIR': [
        'Susceptible',
        'Exposed',
        'Infected',
        'Recovered',
    ], 
    'SEIRD': [
        'Susceptible',
        'Exposed',
        'Infected',
        'Recovered',
        'Deceased',
    ], 
    'SEIRAD': [
        'Susceptible',
        'Exposed',
        'Symptomatic',
        'Asymptomatic',
        'Recovered',
        'Deceased',
    ], 
    'SEIRADH': [
        'Susceptible',
        'Exposed',
        'Recovered',
        'Deceased',
        'Symptomatic',
        'Asymptomatic',
        'Hospitalized',
    ], 
    'SEIRADHV': [
        'Susceptible',
        'Exposed',
        'Recovered',
        'Deceased',
        'Symptomatic',
        'Asymptomatic',
        'Hospitalized',
        'Vaccinated',
    ], 
}

global_labels_map = {
    'Susceptible': 'Susceptible',
    'Exposed': 'Exposed',
    'Infected': 'Infected',
    'Recovered': 'Recovered',
    'Deceased': 'Deceased',
    'Symptomatic': 'Symptomatic',
    'Asymptomatic': 'Asymptomatic',
    'Hospitalized': 'Hospitalized',
    'Vaccinated': 'Vaccinated',
    'Actions': 'Actions',
    'Infected_cumul': 'Cumulative infected',
    'Deceased_cumul': 'Cumulative deceased',
    'Vaccinated_cumul': 'Cumulative vaccinated',
    'Recovered_cumul': 'Cumulative recovered',
    'Infected_rew': 'Infected reward',
    'Hospitalized_rew': 'Hospitalized reward',
    'Deceased_rew': 'Deceased reward',
    'Economy': r'$Cost_{eco}(t)$',
    'Health': r'$Cost_{health}(t)$',
    'Reward': r'$R(t)$',
}


""" We annotate differently based on the environment """
global_annotation_map = {
    'SIR': (['Infected'],[(0.9, 0.4)]), 
    'SEIR': (['Infected','Exposed'],[(0.7, 0.6), (0.8, 0.5)]), 
    'SEIRD': (['Infected','Deceased'],[(0.8, 0.5), (0.9, 0.4)]), 
    'SEIRAD': (['Symptomatic','Deceased'],[(0.4, 0.4), (0.3, 0.6)]), 
    'SEIRADH': (['Hospitalized','Deceased'], [(0.7, 0.5), (0.8, 0.4)]),
    'SEIRADHV': (['Hospitalized','Deceased'], [(0.7, 0.5), (0.8, 0.4)]),
}


def set_plot_env() -> None:
    mpl.rcdefaults()
    mpl.rcParams['mathtext.default']= 'regular'
    mpl.rcParams['font.size'] = 18.
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.weight'] = "normal"
    mpl.rcParams['axes.labelsize'] = 18.
    mpl.rcParams['xtick.labelsize'] = 18
    mpl.rcParams['ytick.labelsize'] = 18
    
    mpl.rcParams['xtick.major.width'] = 0.6
    mpl.rcParams['ytick.major.width'] = 0.6
    mpl.rcParams['axes.linewidth'] = 0.6
    mpl.rcParams['pdf.fonttype'] = 3
    
    mpl.rcParams["xtick.minor.visible"] = "off"
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["xtick.top"] = "off"
    mpl.rcParams["xtick.major.size"] = 8
    mpl.rcParams["xtick.minor.size"] = 5
    
    mpl.rcParams["ytick.minor.visible"] = "off"
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["ytick.major.size"] = 8
    mpl.rcParams["ytick.minor.size"] = 5
    mpl.rcParams["ytick.right"] = "off"

    mpl.rcParams["savefig.facecolor"] = "w"

    # Figures aesthetics
    #sns.set_theme(font_scale=1.5)
    #plt.style.use('seaborn-white') 



""" Used to annotate the maximums """
def annot_max(
    x: pd.Series, 
    y: pd.Series, 
    ax: mpl.axes.Axes = None, 
    place: Tuple[float] = (0.9, 0.5)
    ) -> None:
    
    fc = colors.to_rgba('white')
    ec = colors.to_rgba('black')
    fc = fc[:-1] + (0.5,) # <--- Change the alpha value of facecolor to be 0.5

    ymax = max(y)
    xpos = y[y == ymax].index[0]
    xmax = x[xpos]
    text = "y={:.1f}, x={:.1f}".format(ymax, xmax)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc=fc, ec=ec, alpha=0.5, lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3", color="black")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax),  xytext=place, **kw)


""" 
Dynamics we used in the paper. Similar to 'plot_curves', but this one uses 
some graphic enhancements we could not incorporate in the former function 
(which is used everywhere else).
"""
def plot_env_dynamics(
    plot_data: pd.DataFrame,
    rootdir: str | os.PathLike,
    filename: str = 'compartments_dynamics',
    show_plot: bool = True,
    ) -> None:
    env_name = plot_data['Environment'][0]
    _, axis = plt.subplots(ncols=1, nrows=1, figsize=(10, 8), facecolor="#ffffff")

    # All compartments
    labels = env_labels_map[f'{env_name}']
    p = sns.lineplot(
        data=plot_data[['Days'] + labels],
        ax=axis, 
        dashes=None, 
        linewidth=2, 
        linestyle='-', 
        palette=[dynamics_cmap[f'{l}'] for l in labels]
        )
    p.set_xlabel("Days")

    ax2 = p.secondary_yaxis("right")
    ax2.tick_params(axis="y", direction="in", length=4)
    ax2.set_yticklabels([])

    leg = p.legend(title="Compartments", loc='upper right', fontsize=14, labels=labels)
    for i, l in enumerate(labels):
        leg.legend_handles[i].set_color(dynamics_cmap[f'{l}'])
           

    
    # Set the linewidth of each legend object
    for legobj in leg.legend_handles:
        legobj.set_linestyle('-')
        legobj.set_linewidth(1.5)
    
    #ax = p.twinx()
    #ax.set_yticks([]) # Removes major ticks
    #ax.set_yticklabels([])


    os.makedirs(f"{rootdir}", exist_ok=True)
    filepath = os.path.join(rootdir, f"{filename}.pdf")
    plt.savefig(filepath, pad_inches=0, bbox_inches='tight', transparent=True)
    if show_plot:
        plt.show()
    plt.close()


""" Main routine for plotting the curves """
def plot_curves(
    plot_data: pd.DataFrame, 
    rootdir: str | os.PathLike,
    prefixname: str,
    labels: List[str] = [],
    x_axis: str = 'Days',
    hosp_cap: bool = False,
    seventy_percent_hosp: bool = False,
    span_actions: bool = False,
    separated: bool = False,
    annotate_max: bool = False,
    figsize: Tuple[int, int] = (10, 8),
    facecolor: str = "#ffffff",
    legend_loc: str = "best",
    xlim: float | None = None,
    show_plot: bool = True,
    ) -> None:

    if not 'Environment' in plot_data.columns:
        raise ValueError(f"Invalid dataframe. 'Environment' column not found.")

    env_name = plot_data['Environment'][0]

    if not labels:
        labels = [c for c in plot_data.columns if not c in {
            'Days', 
            'Environment',
            'N',
            'Hosp_cap',
            }]
    else:
        rem = [l for l in labels if l not in plot_data.columns]
        if rem:
            raise ValueError(f"Following column keys were not found in the dataframe '{rem}'. Please \
                use keys corresponding to the environment '{env_name}'")
    
    os.makedirs(rootdir, exist_ok=True)

    if not separated:
        _, axis = plt.subplots(ncols=1, nrows=1, figsize=figsize, facecolor=facecolor)
    
    for lbl in labels:
        if lbl == x_axis:
            continue
        if separated:
            _, axis = plt.subplots(ncols=1, nrows=1, figsize=figsize, facecolor=facecolor)

        p = sns.lineplot(
            y=f"{lbl}", 
            x=f"{x_axis}", 
            data=plot_data[['Days'] + labels], 
            color=dynamics_cmap[f"{lbl}"],
            ax=axis, 
            label=global_labels_map[f"{lbl}"], 
            linewidth=2.5
            ).set(xlabel=x_axis, ylabel=None)

        if annotate_max:
            # Annotating peaks
            for attribute, place in zip(global_annotation_map[env_name][0], global_annotation_map[env_name][1]):
                if place:
                    annot_max(getattr(plot_data, 'Days'), getattr(plot_data, f"{attribute}"), place=place, ax=axis)
                else:
                    annot_max(getattr(plot_data, 'Days'), getattr(plot_data, f"{attribute}"), None, ax=axis)
        
        if (lbl=='Hospitalized'):
            if hosp_cap:
                if not 'Hosp_Cap' in plot_data.columns:
                    raise ValueError(f"Invalid dataframe. 'Hosp_Cap' column not found.")
                max_cap = plot_data['Hosp_Cap'][0]
                axis.axhline(max_cap, color="r", linestyle='--',label=r'$C_{h}$', linewidth=2.5)
                if seventy_percent_hosp:
                    axis.axhline(0.7 * max_cap, color="g", linestyle='-.',label=r"$70\%$ of $C_h$")
        
        if separated:
            axis.legend(loc=f"{legend_loc}")

            # Actions
            if span_actions:
                actions_span(plot_data, axis)
            if xlim:
                plt.xlim(left=0, right=xlim)
            
            # Saving each figure in a different file
            if prefixname:
                filepath = os.path.join(rootdir, f"{prefixname}_{lbl.lower()}.pdf")
            else:
                filepath = os.path.join(rootdir, f"{lbl.lower()}.pdf")
            plt.savefig(filepath, pad_inches=0, bbox_inches='tight', transparent=True)
            if show_plot:
                plt.show()
            plt.close()

    if not separated:
        axis.legend(loc=f"{legend_loc}")

        # Actions
        if span_actions:
            actions_span(plot_data, axis)
        if xlim:
            plt.xlim(left=0, right=xlim)
        # Saving all figures in a single file
        if prefixname:
            filepath = os.path.join(rootdir, f"{prefixname}.pdf")
        else:
            filepath = os.path.join(rootdir, f"all.pdf")
        plt.savefig(filepath, pad_inches=0, bbox_inches='tight', transparent=True)
        if show_plot:
            plt.show()
        plt.close()



""" Plotting Economy, health and rewards """
def plot_cost_reward (
    data: pd.DataFrame,
    rootdir: str | os.PathLike,
    filename: str,
    show_plot: bool = True,
    ) -> None:
    plot_curves(
        plot_data = data, 
        rootdir = rootdir,
        prefixname = filename,
        labels = ['Economy', 'Health', 'Reward'],
        show_plot = show_plot,
        )


""" Plots selected actions """
def plot_actions(
    plot_data: pd.DataFrame,
    rootdir: str | os.PathLike,
    filename: str,
    show_plot: bool = True,
    ) -> None:
    plot_curves(
        plot_data = plot_data, 
        rootdir = rootdir,
        prefixname = filename,
        labels = ['Actions'],
        show_plot = show_plot,
        )


""" Plots selected actions in a piechart format """
def action_piechart(
    plot_data: pd.DataFrame,
    rootdir: str | os.PathLike,
    filename: str,
    show_plot: bool = True,
    ) -> None:
    action_list = plot_data['Actions'].to_list()
    actions_struct = ACTIONS_STRUCT
    _, axis = plt.subplots(ncols=1, nrows=1, figsize=(10, 8), facecolor="#ffffff")
    palette = sns.color_palette(["#2FDD92", "#FFD56B", "#FF5D5D"])
    action_list.sort()
    actions_per_cat = list(Counter(action_list).values())
    perc_actions = np.divide(actions_per_cat,
                             np.sum(actions_per_cat))

    labels = list(Counter(action_list).keys())
    colors = []
    explode = [0]

    for i in range(len(labels)):
        colors.append(palette[labels[i]])
        labels[i] = actions_struct[labels[i]][-1]
        if len(labels) == 2:
            explode = [0, .1]
        else:
            if len(labels) == 3:
                explode = [0, 0, .1]

    wedges, _, autotexts = axis.pie(
        perc_actions,
        colors=colors,
        explode=explode,
        autopct='%1.1f%%',
        shadow=True,
        startangle=90,
        )

    plt.setp(autotexts, size=14, weight="bold")

    axis.legend(wedges,
                labels,
                title="Selected actions",
                bbox_to_anchor=(0.85, 1),
                loc="best")

    os.makedirs(f"{rootdir}", exist_ok=True)
    filepath = os.path.join(rootdir, f"{filename}.pdf")
    plt.savefig(filepath, pad_inches=0, bbox_inches='tight', transparent=True)
    if show_plot:
        plt.show()
    plt.close()


""" 
Uses the same main plotting function. Only difference is this one takes 
dataframes with multiple records to highlight the variance accross 
experiments. 
"""
def plot_with_variance(
    plot_data:pd.DataFrame,
    labels: List[str] = [
        'Infected',
        'Deceased',
        'Hospitalized'
        ],
    rootdir: str | os.PathLike = './outputs',
    filename: str = 'figure',
    show_plot: bool = True,
    ) -> None:
    
    # We always plot these two metrics
    plot_curves(
        plot_data = plot_data, 
        rootdir = rootdir,
        prefixname = f'{filename}_cost_reward' if filename else 'cost_reward',
        labels = ['Economy', 'Actions'],
        show_plot = show_plot,
    )

    # Then the rest
    env_name = plot_data[f"Environment"][0]
    plot_curves(
        plot_data = plot_data, 
        rootdir = rootdir,
        prefixname = f'{filename}_inf_dec_hosp' if filename else 'inf_dec_hosp',
        labels = [x for x in labels if x in env_labels_map[f"{env_name}"]] ,
        hosp_cap = True,
        annotate_max = True,
        show_plot = show_plot,
    )


""" Plots actions in background """
def actions_span(
    plot_data: pd.DataFrame,  
    axis: mpl.axes.Axes,
    best_loc: str='lower right',
    ) -> None:
    actions = plot_data['Actions'].to_list()
    actions_struct = ACTIONS_STRUCT
    sorted_actions = sorted(actions)
    handles = []
    handels_label = []
    pst = actions[0]
    pos = 0
    c = ["#2FDD92", "#FFD56B", "#FF5D5D"]
    dc = [0, 0, 0]

    a = list(Counter(sorted_actions).keys())
    matrix = {}
    for i in range(len(a)):
        matrix[a[i]] = [c[a[i]], actions_struct[a[i]][-1]]

    for i in range(1, len(actions)):
        if actions[i] != pst:
            axis.axvspan(pos+1, i+1, alpha=0.4, color=matrix[pst][0])
            if matrix[pst][1] not in handels_label:
                handles.append(mpatches.Patch(
                    color=matrix[pst][0], label=matrix[pst][1], alpha=0.6))
                handels_label.append(matrix[pst][1])
            dc[pst] += 1
            pos, pst = i, actions[i]

    axis.axvspan(pos+1, len(actions), alpha=0.4,
                 color=matrix[pst][0],
                 label='_'*dc[pst] + matrix[pst][1])
    if matrix[pst][1] not in handels_label:
        handles.append(mpatches.Patch(
            color=matrix[pst][0], label=matrix[pst][1], alpha=0.6))
        handels_label.append(matrix[pst][1])
    leg = Legend(
        axis, 
        handles, 
        handels_label, 
        loc=best_loc, 
        fancybox=True, 
        framealpha=0.5
        )
    axis.add_artist(leg)



""" Focus on hospitalizations """
def plot_hospitalized_w_threshold(
    plot_data: pd.DataFrame, 
    rootdir: str | os.PathLike,
    filename: str,
    show_plot: bool = True,
    ) -> None:
    plot_curves(
        plot_data=plot_data, 
        rootdir=rootdir,
        prefixname=filename,
        labels = ['Hospitalized'],
        hosp_cap = True,
        seventy_percent_hosp = True,
        show_plot = show_plot,
    )


""" Focus on infections/deaths """
def plot_health_critical_metrics(
    plot_data: pd.DataFrame, 
    rootdir: str | os.PathLike,
    filename: str,
    show_plot: bool = True,
    ) -> None:

    plot_curves(
        plot_data=plot_data, 
        rootdir=rootdir,
        prefixname=filename,
        labels = ['Infected', 'Deceased'],
        show_plot = show_plot,
    )


""" 
Follows a specific separation scheme, otherwise mainly uses the 'plot_curves'
routine with different arguments.
"""
def plot_categorized_dynamics(
    plot_data: pd.DataFrame, 
    rootdir: str | os.PathLike,
    filename: str,
    show_plot: bool = True,
    ) -> None:
    
    env_name = plot_data['Environment'][0]

    # Susceptible & vaccinated
    plot_curves(
        plot_data=plot_data, 
        rootdir=rootdir,
        prefixname=filename,
        labels = [x for x in ['Susceptible', 'Vaccinated'] if x in env_labels_map[f"{env_name}"]],
        show_plot = show_plot, 
    )

    # Hospitalized, deceased & recovered
    plot_curves(
        plot_data=plot_data, 
        rootdir=rootdir,
        prefixname=filename,
        labels = [x for x in ['Hospitalized', 'Deceased', 'Recovered'] if x in env_labels_map[f"{env_name}"]],
        show_plot = show_plot,  
    )


    # Infected & exposed
    env_name = plot_data['Environment'][0]
    plot_curves(
        plot_data=plot_data, 
        rootdir=rootdir,
        prefixname=filename,
        labels = [x for x in ['Infected', 'Exposed', 'Symptomatic', 'Asymptomatic'] if x in env_labels_map[f"{env_name}"]], 
        show_plot = show_plot, 
    )


""" Plots compartments dynamics, costs and rewards and agent's actions """
def plot_env_metrics(
    plot_data: pd.DataFrame,
    rootdir:str | os.PathLike = "./outputs/figures",
    filename: str = "metrics",
    show_plot: bool = True,
    ) -> None:

    env_name = plot_data["Environment"][0]
    
    # Plotting dynamics
    plot_env_dynamics(plot_data, rootdir, f"{filename}_dynamics", show_plot)

    # Same, but grouped by category
    plot_categorized_dynamics(plot_data, rootdir, f"{filename}_cat_dynamics", show_plot)

    if env_name in {'SEIRADH', 'SEIRADHV'}:
        # Hopsitalized 
        plot_hospitalized_w_threshold(plot_data, rootdir, f"{filename}_hosp_tresholds")


        # Health rewards
        plot_curves(
            plot_data = plot_data, 
            rootdir = rootdir,
            prefixname = f"{filename}_cost_rewards",
            labels = ['Infected_rew', 'Hospitalized_rew', 'Deceased_rew', 'Health'],
            show_plot = show_plot
        )

    if env_name == 'SEIRADHV':
        # Cumulative infections/deaths
        plot_cum_critic_metrics(plot_data, rootdir, f"{filename}_cum_critics", show_plot)

    # Plotting other metrics
    plot_cost_reward(plot_data, rootdir, f"{filename}_cost_reward", show_plot)

    # Actions
    plot_actions(plot_data, rootdir, f"actions", show_plot)

    # Actions in a piechart
    action_piechart(plot_data, rootdir, f"piechart", show_plot)
    

""" Plots cummulative infections, deaths, recoveries and vaccinations """
def plot_cum_critic_metrics(
    plot_data: pd.DataFrame,
    rootdir: str | os.PathLike,
    filename: str,
    show_plot: bool = True,
    ) -> None:

    plot_curves(
        plot_data=plot_data, 
        rootdir=rootdir,
        prefixname=f"{filename}_inf_dec_cum",
        labels = ['Infected_cumul', 'Deceased_cumul'],
        legend_loc="upper left",
        span_actions=True,
        show_plot=show_plot,
    )

    plot_curves(
        plot_data=plot_data, 
        rootdir=rootdir,
        prefixname=f"{filename}_rec_vac_cum" if filename else f"rec_vac_cum",
        labels = ['Recovered_cumul', 'Vaccinated_cumul'],
        legend_loc="upper left",
        span_actions=True,
        show_plot=show_plot,
    )


""" Plots predicted curves against actual ones """
def compare_metrics(
    predicted: pd.DataFrame, 
    actual: pd.DataFrame, 
    rootdir: str | os.PathLike,
    filename: str ='comparison', 
    label_map: Dict[str, str] = {
        'Hospitalized': 'current_hospitalized_patients',
        'Deceased': 'cum_deceased',
    },
    show_plot: bool = True,
    ) -> None:

    ### All labels
    for label in label_map.keys():
        _, axis = plt.subplots(ncols=1, nrows=1, figsize=(10, 8), facecolor="#ffffff")

        # Results of agent's management
        sns.lineplot(x='Days', y=label, data=predicted,  color=dynamics_cmap[label], ax=axis, label=label, linewidth=2.5).set(xlabel="Days", ylabel=None)
        # Actual numbers
        sns.lineplot(x='Days', y=label_map[label], data=actual,  color=dynamics_cmap[f"{label}_ref"], ax=axis, label=f"{label} (actual)", linewidth=2.5).set(xlabel="Days", ylabel=None)
        axis.set_yscale('log')
        axis.legend(loc="best")
 
        if label == "Hospitalized":
            if not 'Hosp_Cap' in predicted.columns:
                raise ValueError(f"Invalid dataframe. 'Hosp_Cap' column not found.")
            max_cap = predicted['Hosp_Cap'][0]
            axis.axhline(max_cap, color="r", linestyle='--',label=r'$C_{h}$', linewidth=2.5)
        plt.xlim(left=0, right=min(len(predicted, actual)))

        os.makedirs(f"{rootdir}", exist_ok=True)
        filepath = os.path.join(f"{rootdir}", f"{filename}_{label}.pdf" if filename else f"comparison_{label}.pdf")
        plt.savefig(f"{filepath}", pad_inches=0, bbox_inches='tight', transparent=True)
        if show_plot:
            plt.show()
        plt.close()


""" Used to plot heatmaps """
def plot_heatmap(
    data: pd.DataFrame,
    rootdir: str | os.PathLike,
    filename: str,
    figsize: Tuple[int, int] = (10,8), 
    facecolor: str ="#ffffff",
    color_map: Literal['RdPu', 'viridis_r', 'YlGnBu', 'magma_r'] = 'RdPu',
    xlabel: str | None = None, 
    ylabel: str | None = None,
    title: str | None = None,
    vmin: float = 0.,
    vmax: float = .26,
    show_plot: bool = True,
    ):

    _, axis = plt.subplots(1, 1, figsize=figsize, facecolor=f"{facecolor}")  
    p = sns.heatmap(data.T, vmin=vmin, vmax=vmax, cmap=f"{color_map}", ax=axis)
    if xlabel:
        p.set(xlabel=f"{xlabel}")
    if ylabel:
        p.set(ylabel=f"{ylabel}")
    if title:
        p.set(title=f"{title}")

    # Specific to heatmaps using population sizes
    if (ylabel == 'Population size'):
        xlabels=[]
        for y in p.get_yticklabels():
            y_int = int(y.get_text())
            a = y_int//1e3
            if (y_int > int(5e4)):
                a = y_int//1e6
                xlabels += ['{a:.1f}M'.format(a=a)]
            else:
                xlabels += ['{a:.1f}K'.format(a=a)]
        p.set_yticklabels(xlabels)

    
    os.makedirs(f"{rootdir}", exist_ok=True)
    filepath = os.path.join(f"{rootdir}", f"{filename}.pdf")
    plt.savefig(f"{filepath}", pad_inches=0, bbox_inches='tight', transparent=True)
    if show_plot:
        plt.show()
    plt.close()

