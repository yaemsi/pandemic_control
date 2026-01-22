import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

from loguru import logger

from pandemic_control.utils import (
    run_simulation_const_policy,
    plot_env_dynamics,
    plot_env_metrics,
    plot_with_variance,
    plot_actions,
    run_simulation,
    run_n_simulations,
    run_env_sim,
    plot_heatmap,
    RewardCallback
)
from pandemic_control.model import (
    RLModel,
)
from pandemic_control.environment import(
    SIR_Env,
    SEIR_Env,
    SEIRD_Env,
    SEIRAD_Env,
    SEIRADH_Env,
    SEIRADHV_Env
)

def run_const_policy_simulations(args: argparse.Namespace) -> None:
    logger.info(f"*** Loading '{args.env_type}' environment.\n")
    CLS = getattr(sys.modules[__name__], f"{args.env_type}_Env")
    env = CLS(args.cfg_file)
    env.reset(seed=args.seed)
    for mode in ['no_restr', 'soc_dist', 'lockdown', 'random']:
        logger.info(f"*** Running simulation with '{mode}' policy...")
        data = run_simulation_const_policy(
            env=env, 
            t_max=args.t_max, 
            mode=mode, 
            seed=args.seed
            )
        data = pd.DataFrame(data)
        logger.info(f"*** Generating plots...")
        plot_env_dynamics(
            plot_data=data,
            #rootdir=f"{args.output_dir}/basic_tests/const_policy/figures",
            rootdir=f"{args.output_dir}",
            filename=f"{args.env_type.lower()}_{mode.lower()}_dynamics",
        )
        logger.info(f"*** Done.\n\n")

def run_model_policy_simulations(args: argparse.Namespace) -> None:
    logger.info(f"*** Loading '{args.env_type}' environment.")
    CLS = getattr(sys.modules[__name__], f"{args.env_type}_Env")
    env = CLS(args.cfg_file)
    env.reset(seed=args.seed)

    #rootdir=f"./outputs/basic_tests/rl_model"
    rootdir=f"{args.output_dir}"
    logger.info(f"*** Loading '{args.model_type}' agent.")
    #output_dir = f"./outputs/basic_tests/rl_model/saved_weights/{args.env_type.lower()}_{args.model_type.lower()}"
    output_dir = f"{args.output_dir}/saved_weights/{args.env_type.lower()}_{args.model_type.lower()}"

    model = RLModel.from_metadata (
        env = env,
        model_type = f"{args.model_type}",
        device = f"cpu",
        seed = args.seed,
        verbose = 1,
        tensorboard_log = f"./outputs/basic_tests/rl_model/logs",
        n_steps = args.t_max//7, #TODO
    )
    
    logger.info(f"*** Starting training.")
    cb = RewardCallback()
    model.train(
        timesteps = args.timesteps,
        output_dir = output_dir,
        tb_log_name = f"{args.env_type.lower()}_history",
        epochs = args.epochs,
        save_interval = args.save_interval,
        callback = cb,
    )
    logger.info (f"################### cb.counter == {cb.counter}")
    logger.info (f"################### len(cb.episode_rewards) == {len(cb.episode_rewards)}")
    logger.info(f"*** Loading model from disk.")
    model = RLModel.load_from_disk(
        model_type = f"{args.model_type}",
        model_weights = os.path.join(f"{output_dir}", f"final", f"model.bin"),
        device=f"cpu",
    )

    logger.info(f"*** Running one simulation.")

    logger.info(f"*** Plotting curves based on one years' data")
    data = run_simulation(
        env = env,
        model = model, 
        t_max = args.t_max,
    )
    data = pd.DataFrame(data)
    plot_env_dynamics(
        plot_data=data.head(args.t_max),
        rootdir=f"{rootdir}/figures", 
        filename=f"{args.env_type.lower()}_{args.model_type.lower()}_dynamics"
    )

    plot_env_metrics(
        plot_data=data,
        rootdir=f"{rootdir}/figures", 
        filename=f"metrics_{args.env_type.lower()}_{args.model_type.lower()}",
    )
    
    logger.info(f"*** Running {args.rounds} simulations.")
    data = run_n_simulations(
        env = env, 
        model = model, 
        rounds = args.rounds,
        t_max = args.t_max,
        )
    logger.info(f"*** Simulations complete. Generating variance plots.")
    data = pd.DataFrame(data)
    
    # Compartments
    plot_with_variance(
        plot_data=data,
        rootdir=f"{rootdir}/figures",
        filename=f"dynamics_variance"
        )
    
    # Actions
    plot_actions(
        plot_data=data, 
        rootdir=f"{rootdir}/figures",
        filename=f"actions_variance",
        )

    logger.info(f"*** Simulations completed.")

def plot_heatmaps_pop_size(args: argparse.Namespace) -> None:
    filepath = f"{args.cfg_file}"
    rootdir = f"{args.output_dir}/heatmaps"
    with open(filepath, 'r') as fp:
        config = json.load(fp)

    """
    Deaths, hospitalizations and infections as functions of population size
    """
    logger.info(
        f"*** Plotting deaths, hospitalizations and infections as functions \
        of population size."
        )
    pop_sizes = [500, 5000, 50000, 500000, 5000000, 50000000, 500000000]
    hosp_caps = [15 ,300, 300, 300, 3000, 3000, 100000]

    dth_data_size = {}
    inf_data_size = {}
    hsp_data_size = {}

    for N, hosp_cap in zip(pop_sizes, hosp_caps):
        config[f'N'] = N
        config[f'hosp_cap'] = hosp_cap
        CLS = getattr(sys.modules[__name__], f"{args.env_type}_Env")
        env = CLS(args.cfg_file)
        env.reset(seed=args.seed)
        #env = SEIRADHV_Env(config)

        run_data = run_env_sim(
            env = env,  
            t_max = config['env-params']['max_steps'], 
            mode = "Test",
            selected_action = 0
            )
        
        dth_data_size[f'{N}'] = np.array(run_data['Deceased'])/N
        inf_data_size[f'{N}'] = np.array(run_data['Infected'])/N
        hsp_data_size[f'{N}'] = np.array(run_data['Hospitalized'])/N


    plot_heatmap(
        data = pd.DataFrame(dth_data_size),
        rootdir = f"{rootdir}", 
        filename = f"pop_size_deaths",
        color_map = "magma_r",
        xlabel = f"Days", 
        ylabel = f"Population size",
        title = f"Deaths rate",
        vmin=0, 
        vmax=.066, 
        )

    plot_heatmap(
        data = pd.DataFrame(inf_data_size), 
        rootdir = f"{rootdir}", 
        filename = f"pop_size_infections",
        color_map = "RdPu",
        xlabel = f"Days", 
        ylabel = f"Population size",
        title = f"Infections rate",
        vmin=0, 
        vmax=.26,
        )

    plot_heatmap(
        data = pd.DataFrame(hsp_data_size), 
        rootdir = f"{rootdir}", 
        filename = f"pop_size_hospitalizations",
        color_map = "viridis_r",
        xlabel = f"Days", 
        ylabel = f"Population size",
        title = f"Hospitalizations rate",
        vmin=0, 
        vmax=.115,
        )
    logger.info(f"*** Done.")


def plot_heatmaps_ph_pd(args: argparse.Namespace) -> None:
    filepath = f"{args.cfg_file}"
    rootdir = f"{args.output_dir}/heatmaps"
    with open(filepath, 'r') as fp:
        config = json.load(fp)
    """
    Deaths, hospitalizations, infections, cummulative hospitalizations 
    and cummulative infections as functions of population resistance
    """
    logger.info(
        f"*** Plotting deaths, hospitalizations, infections, cummulative \
        hospitalizations and cummulative infections as functions of \
        population resistance."
        )
    phs = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
    pds = [0.01, 0.03 , 0.05, 0.08, 0.1, 0.11, 0.15, 0.2, 0.25, 0.3]


    dth_data_type_max = {}
    inf_data_type_max = {}
    hsp_data_type_max = {}
    inf_data_type_cumul = {}
    hsp_data_type_cumul = {}

    for p_h in phs:
        dth_max = []
        hsp_max = []
        inf_max = []
        
        inf_cumul = []
        hsp_cumul = []
        for p_d in pds:
            config["spec-params"]["probas"] =  [
                0.8,    # Probability of symptomatic covid
                p_h,    # Probability of hospitalization
                p_d,    # Probability of death in hospital
                0.0727  # Probability of death
                ] 

            CLS = getattr(sys.modules[__name__], f"{args.env_type}_Env")
            env = CLS(args.cfg_file)
            env.reset(seed=args.seed)
            run_data = run_env_sim(
                env=env,  
                t_max = config['env-params']['max_steps'], 
                mode = "Test",
                selected_action = 0
                )
                
            #append max
            N = config['env-params']['N']
            inf_max.append(max(np.array(run_data['Infected']))/N)
            hsp_max.append(max(np.array(run_data['Hospitalized']))/N)
            dth_max.append(max(np.array(run_data['Deceased']))/N)
                
            #append cumul
            inf_cumul.append(run_data['Infected_cumul'][-1]/N)
            hsp_cumul.append(run_data['Hospitalized_cumul'][-1]/N)
            
        inf_data_type_max[f'{p_h}'] = inf_max
        hsp_data_type_max[f'{p_h}'] = hsp_max
        dth_data_type_max[f'{p_h}'] = dth_max
        inf_data_type_cumul[f'{p_h}'] = inf_cumul
        hsp_data_type_cumul[f'{p_h}'] = hsp_cumul


    # plotting heatmaps
    plot_heatmap(
        data = pd.DataFrame(inf_data_type_max, index=pds),
        rootdir = f"{rootdir}",
        filename = f"infections",
        color_map = "RdPu",
        xlabel="Death probability", 
        ylabel="Hospitalization probability", 
        title="Infection peaks",
        vmin= 0,
        vmax= 1,
        )

    plot_heatmap(
        data = pd.DataFrame(hsp_data_type_max, index=pds),
        rootdir = f"{rootdir}",
        filename = f"hospitalizations",
        color_map = "viridis_r",
        xlabel="Death probability", 
        ylabel="Hospitalization probability", 
        title="Hospitalization peaks",
        vmin= 0.,
        vmax= .4,
        )

    plot_heatmap(
        data = pd.DataFrame(dth_data_type_max, index=pds),
        rootdir = f"{rootdir}",
        filename = f"deaths",
        color_map = "magma_r",
        xlabel="Death probability", 
        ylabel="Hospitalization probability", 
        title="Death peaks",
        vmin= 0.,
        vmax= .5,
        )


    plot_heatmap(
        data = pd.DataFrame(inf_data_type_cumul , index=pds),
        rootdir = f"{rootdir}",
        filename = f"cum_infections",
        color_map = "viridis_r",
        xlabel="Death probability", 
        ylabel="Hospitalization probability", 
        title="Cummulative infections",
        vmin= 0.,
        vmax= 1.,
        )


    plot_heatmap(
        data = pd.DataFrame(hsp_data_type_cumul , index=pds),
        rootdir = f"{rootdir}",
        filename = f"cum_hospitalizations",
        color_map = "YlGnBu",
        xlabel="Death probability", 
        ylabel="Hospitalization probability", 
        title="Cummulative hospitalizations",
        vmin= 0.,
        vmax= 0.8,
        )
    
    logger.info(f"*** Done.")


def plot_heatmaps_omega_rho(args: argparse.Namespace) -> None:
    filepath = f"{args.cfg_file}"
    rootdir = f"{args.output_dir}/heatmaps"
    with open(filepath, 'r') as fp:
        config = json.load(fp)
    """
    Deaths, hospitalizations, infections, vaccinations, cummulative 
    hospitalizations and cummulative infections as functions of vaccination 
    rate/inefficiency
    """
    logger.info(
        f"*** Deaths, hospitalizations, infections, vaccinations, cummulative \
        hospitalizations and cummulative infections as functions of vaccination \
        rate/inefficiency."
        )
    # OMEGA vs RHO rates
    omegas = [0.02, 0.018, 0.016, 0.014, 0.012, 0.01, 0.008, 0.005, 0.003, 0.001] # Vaccine rate
    rhos = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] # Vaccine inefficiency

    dth_data_vax_max = {}
    inf_data_vax_max = {}
    hsp_data_vax_max = {}
    vax_data_vax_max = {}
    inf_data_vax_cumul = {}
    hsp_data_vax_cumul = {}

    for omega in omegas:
        dth_max = []
        hsp_max = []
        inf_max = []
        vax_max = []
        
        inf_cumul = []
        hsp_cumul = []
        for rho in rhos:
            config["spec-params"]["omega"] = omega
            config["spec-params"]["rho"] = rho

            CLS = getattr(sys.modules[__name__], f"{args.env_type}_Env")
            env = CLS(args.cfg_file)
            env.reset(seed=args.seed)
            run_data = run_env_sim(
                env=env,  
                t_max = config['env-params']['max_steps'], 
                mode = "Test",
                selected_action = 0
                )
            
            #append max
            N = config['env-params']['N']
            inf_max.append(max(np.array(run_data['Infected']))/N)
            hsp_max.append(max(np.array(run_data['Hospitalized']))/N)
            dth_max.append(max(np.array(run_data['Deceased']))/N)
            vax_max.append(run_data['Vaccinated'][-1]/N)
            
            #append cumul
            inf_cumul.append(run_data['Infected_cumul'][-1]/N)
            hsp_cumul.append(run_data['Hospitalized_cumul'][-1]/N)
            
        inf_data_vax_max[f'{omega}'] = inf_max
        hsp_data_vax_max[f'{omega}'] = hsp_max
        dth_data_vax_max[f'{omega}'] = dth_max
        vax_data_vax_max[f'{omega}'] = vax_max
        
        inf_data_vax_cumul[f'{omega}'] = inf_cumul
        hsp_data_vax_cumul[f'{omega}'] = hsp_cumul


    plot_heatmap(
        data = pd.DataFrame(inf_data_vax_max, index=rhos).T,
        rootdir = f"{rootdir}",
        filename = f"peak_infections",
        color_map = 'RdPu',
        vmin = .0,
        vmax = .4,
        xlabel="Vaccine inefficacy rate",
        ylabel="Vaccination rate",
        title="Infections peak",
        )


    plot_heatmap(
        data = pd.DataFrame(hsp_data_vax_max, index=rhos).T,
        rootdir = f"{rootdir}",
        filename = f"peak_hospitalizations",
        color_map = 'viridis_r',
        vmin = .03,
        vmax = .3,
        xlabel="Vaccine inefficacy rate",
        ylabel="Vaccination rate",
        title="Hospitalizations peak",
        )

    plot_heatmap(
        data = pd.DataFrame(vax_data_vax_max, index=rhos).T,
        rootdir = f"{rootdir}",
        filename = f"peak_vaccinations",
        color_map = 'RdPu',
        vmin = 0.,
        vmax = 1.,
        xlabel="Vaccine inefficacy rate",
        ylabel="Vaccination rate",
        title="Vaccination peak",
        )

    plot_heatmap(
        data = pd.DataFrame(dth_data_vax_max, index=rhos).T,
        rootdir = f"{rootdir}",
        filename = f"peak_deaths",
        color_map = 'magma_r',
        vmin = .0,
        vmax = .115,
        xlabel="Vaccine inefficacy rate",
        ylabel="Vaccination rate",
        title="Death rate",
        )

    plot_heatmap(
        data = pd.DataFrame(inf_data_vax_cumul, index=rhos).T,
        rootdir = f"{rootdir}",
        filename = f"cum_vaccinations",
        color_map = 'YlOrBr',
        vmin = .5,
        vmax = 2.,
        xlabel="Vaccine inefficacy rate",
        ylabel="Vaccination rate",
        title="Cummulative infections",
        )


    plot_heatmap(
        data = pd.DataFrame(hsp_data_vax_cumul, index=rhos).T,
        rootdir = f"{rootdir}",
        filename = f"cum_hospitalizations",
        color_map = 'viridis_r',
        vmin = .0,
        vmax = .7,
        xlabel="Vaccine inefficacy rate",
        ylabel="Vaccination rate",
        title="Cummulative hospitalizations",
        )
    
    logger.info(f"*** Done.")
    

def run_plot_heatmaps(args: argparse.Namespace) -> None:
    #filepath = "D:\\Projects\\pandemic_control\\configs\\envs_tests\\default-seiradhv.json"
    logger.info(f"*** Plotting heatmaps using environment '{args.env_type}'.")
    plot_heatmaps_pop_size(args)
    plot_heatmaps_ph_pd(args)
    plot_heatmaps_omega_rho(args)
    logger.info(f"*** Heatmaps plotting completed.")
    
    



def plot_learning_curves(args: argparse.Namespace) -> None:
    pass

def run_preprocess_data(args: argparse.Namespace) -> None:
    pass

def run_train(args: argparse.Namespace) -> None:
    pass

def run_predict(args: argparse.Namespace) -> None:
    pass