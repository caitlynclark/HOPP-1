import os
import sys
sys.path.append('')
from dotenv import load_dotenv
import pandas as pd
# import json
from hybrid.sites import SiteInfo
from hybrid.sites import flatirons_site as sample_site
from hybrid.keys import set_developer_nrel_gov_key
from examples.H2_Analysis.hopp_for_Nome import hopp_for_Nome
# from examples.H2_Analysis.run_h2a import run_h2a as run_h2a
from examples.H2_Analysis.simple_dispatch import SimpleDispatch
# from examples.H2_Analysis.simple_cash_annuals import simple_cash_annuals
from examples.H2_Analysis.simple_generator_diesel import SimpleDiesel
import examples.H2_Analysis.check_load_met as check_load_met #import check_load_met


import plot_results
import examples.H2_Analysis.run_h2_PEM as run_h2_PEM
from tools.resource import *
from tools.resource.resource_loader import site_details_creator
import numpy as np
import numpy_financial as npf
# from lcoe.lcoe import lcoe as lcoe_calc
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

def establish_save_output_dict():
    """
    Establishes and returns a 'save_outputs_dict' dict
    for saving the relevant analysis variables for each site.
    """

    save_outputs_dict = dict()
    
    return save_outputs_dict

def save_the_things():
    save_outputs_dict['Site Name'].append(site_name)
    save_outputs_dict['Scenario Choice'].append(scenario_choice)
    save_outputs_dict['Site Lat'].append(lat)
    save_outputs_dict['Site Lon'].append(lon)
    save_outputs_dict['ATB Year'].append(atb_year)
    save_outputs_dict['Resource Year'].append(resource_year)
    save_outputs_dict['Turbine Model'].append(turbine_name)
    save_outputs_dict['Critical Load Factor'].append(critical_load_factor)
    save_outputs_dict['kW continuous load'].append(kw_continuous)
    save_outputs_dict['Useful Life'].append(useful_life)
    save_outputs_dict['PTC'].append(ptc_avail)
    save_outputs_dict['ITC'].append(itc_avail)
    save_outputs_dict['Discount Rate'].append(discount_rate)
    save_outputs_dict['Debt Equity'].append(debt_equity_split)
    save_outputs_dict['Hub Height (m)'].append(tower_height)
    save_outputs_dict['Storage Enabled'].append(storage_used)
    save_outputs_dict['Wind Cost kW'].append(wind_cost_kw)
    save_outputs_dict['Solar Cost kW'].append(solar_cost_kw)
    save_outputs_dict['Storage Cost kW'].append(storage_cost_kw)
    save_outputs_dict['Storage Cost kWh'].append(storage_cost_kwh)
    save_outputs_dict['Storage Hours'].append(storage_hours)
    save_outputs_dict['Wind MW built'].append(wind_size_mw)
    save_outputs_dict['Solar MW built'].append(solar_size_mw)
    save_outputs_dict['Storage MW built'].append(storage_size_mw)
    save_outputs_dict['Storage MWh built'].append(storage_size_mwh)
    save_outputs_dict['Battery Can Grid Charge'].append(battery_can_grid_charge)
    save_outputs_dict['Built Interconnection Size'].append(hybrid_plant.interconnect_kw)
    save_outputs_dict['Total Installed Cost $(HOPP)'].append(total_hopp_installed_cost)
    save_outputs_dict['Total Yearly Electrical Output'].append(total_elec_production)
    save_outputs_dict['LCOE'].append(lcoe)
    save_outputs_dict['Grid Connected HOPP'].append(grid_connected_hopp)
    save_outputs_dict['HOPP Total Generation'].append(np.sum(hybrid_plant.grid.generation_profile[0:8759]))
    save_outputs_dict['Wind Capacity Factor'].append(hybrid_plant.wind._system_model.Outputs.capacity_factor)
    save_outputs_dict['HOPP Energy Shortfall'].append(np.sum(energy_shortfall_hopp))
    save_outputs_dict['HOPP Curtailment'].append(np.sum(combined_pv_wind_curtailment_hopp))
    save_outputs_dict['Battery Generation'].append(np.sum(battery_used))
    save_outputs_dict['Electricity to Grid'].append(np.sum(excess_energy))
    save_outputs_dict['Electrolyzer Stack Size'].append(H2A_Results['electrolyzer_size'])
    save_outputs_dict['Electrolyzer Total System Size'].append(H2A_Results['total_plant_size'])
    return save_outputs_dict
def save_the_things_alt():
    save_outputs_dict['Site Name'] = (site_name)
    save_outputs_dict['Scenario Choice'] = (scenario_choice)
    save_outputs_dict['Site Lat'] = (lat)
    save_outputs_dict['Site Lon'] = (lon)
    save_outputs_dict['ATB Year'] = (atb_year)
    save_outputs_dict['Resource Year'] = (resource_year)
    save_outputs_dict['Turbine Model'] = (turbine_name)
    save_outputs_dict['Critical Load Factor'] = (critical_load_factor)
    save_outputs_dict['kW continuous load'] = (kw_continuous)
    save_outputs_dict['Useful Life'] = (useful_life)
    save_outputs_dict['PTC'] = (ptc_options)
    save_outputs_dict['ITC'] = (itc_avail)
    save_outputs_dict['Discount Rate'] = (discount_rate)
    save_outputs_dict['Debt Equity'] = (debt_equity_split)
    save_outputs_dict['Hub Height (m)'] = (tower_height)
    save_outputs_dict['Storage Enabled'] = (storage_used)
    save_outputs_dict['Wind Cost kW'] = (wind_cost_kw)
    save_outputs_dict['Solar Cost kW'] = (solar_cost_kw)
    save_outputs_dict['Storage Cost kW'] = (storage_cost_kw)
    save_outputs_dict['Storage Cost kWh'] = (storage_cost_kwh)
    save_outputs_dict['Storage Hours'] = (storage_hours)
    save_outputs_dict['Wind MW built'] = (wind_size_mw)
    save_outputs_dict['Solar MW built'] = (solar_size_mw)
    save_outputs_dict['Storage MW built'] = (storage_size_mw)
    save_outputs_dict['Storage MWh built'] = (storage_size_mwh)
    save_outputs_dict['Battery Can Grid Charge'] = (battery_can_grid_charge)
    save_outputs_dict['Built Interconnection Size'] = (hybrid_plant.interconnect_kw)
    save_outputs_dict['Total Installed Cost $(HOPP)'] = (total_hopp_installed_cost)
    save_outputs_dict['Total Yearly Electrical Output'] = (total_elec_production)
    save_outputs_dict['LCOE'] = (lcoe)
    save_outputs_dict['Grid Connected HOPP'] = (grid_connected_hopp)
    save_outputs_dict['HOPP Total Generation'] = (np.sum(hybrid_plant.grid.generation_profile[0:8759]))
    save_outputs_dict['Wind Capacity Factor'] = (hybrid_plant.wind._system_model.Outputs.capacity_factor)
    save_outputs_dict['HOPP Energy Shortfall'] = (np.sum(energy_shortfall_hopp))
    save_outputs_dict['HOPP Curtailment'] = (np.sum(combined_pv_wind_curtailment_hopp))
    save_outputs_dict['Battery Generation'] = (np.sum(battery_used))
    save_outputs_dict['Electricity to Grid'] = (np.sum(excess_energy))
    save_outputs_dict['Percentage Time 50% Load met without storage'] = (perc_time_load_met_no_storage_50)
    save_outputs_dict['Percentage Time 75% Load met without storage'] = (perc_time_load_met_no_storage_75)
    save_outputs_dict['Percentage Time 90% Load met without storage'] = (perc_time_load_met_no_storage_90)
    save_outputs_dict['Percentage Time 95% Load met without storage'] = (perc_time_load_met_no_storage_95)  
    save_outputs_dict['Percentage Time 100% Load met without storage'] = (perc_time_load_met_no_storage_100)
    save_outputs_dict['Percentage Time 50% Load met with storage'] = (perc_time_load_met_with_storage_50)
    save_outputs_dict['Percentage Time 75% Load met with storage'] = (perc_time_load_met_with_storage_75)
    save_outputs_dict['Percentage Time 90% Load met with storage'] = (perc_time_load_met_with_storage_90)
    save_outputs_dict['Percentage Time 95% Load met with storage'] = (perc_time_load_met_with_storage_95)  
    save_outputs_dict['Percentage Time 100% Load met with storage'] = (perc_time_load_met_with_storage_100)

    save_outputs_dict['Percentage Time 50% Load met without diesel'] = (perc_time_load_met_no_storage_50)
    save_outputs_dict['Percentage Time 75% Load met without diesel'] = (perc_time_load_met_no_storage_75)
    save_outputs_dict['Percentage Time 90% Load met without diesel'] = (perc_time_load_met_no_storage_90)
    save_outputs_dict['Percentage Time 95% Load met without diesel'] = (perc_time_load_met_no_storage_95)  
    save_outputs_dict['Percentage Time 100% Load met without diesel'] = (perc_time_load_met_no_storage_100)
    save_outputs_dict['Percentage Time 50% Load met with diesel'] = (perc_time_load_met_with_storage_50)
    save_outputs_dict['Percentage Time 75% Load met with diesel'] = (perc_time_load_met_with_storage_75)
    save_outputs_dict['Percentage Time 90% Load met with diesel'] = (perc_time_load_met_with_storage_90)
    save_outputs_dict['Percentage Time 95% Load met with diesel'] = (perc_time_load_met_with_storage_95)  
    save_outputs_dict['Percentage Time 100% Load met with diesel'] = (perc_time_load_met_with_storage_100)

    return save_outputs_dict

# Establish API Key
load_dotenv()
NREL_API_KEY = os.getenv("NREL_API_KEY")
set_developer_nrel_gov_key('NREL_API_KEY')  # Set this key manually here if you are not setting it using the .env

# Establish Resource Details
resource_year = 2018 #2013
desired_lats = 64.503889
desired_lons = -165.399444
resource_dir = Path(__file__).parent.parent.parent / "resource_files"
sitelist_name = 'filtered_site_details_Nome'
load_resource_from_file = False

if load_resource_from_file:
    print('Downloading Resource from file')
    # Loads resource files in 'resource_files', finds nearest files to 'desired_lats' and 'desired_lons'
    site_details = resource_loader_file(resource_dir, desired_lats, desired_lons, resource_year)  # Return contains
    site_details.to_csv(os.path.join(resource_dir, sitelist_name))
    site_details = filter_sites(site_details, location='usa only')
else:
    # Creates the site_details file containing grid of lats, lons, years, and wind and solar filenames (blank
    # - to force API resource download)
    if os.path.exists(sitelist_name):
        site_details = pd.read_csv(sitelist_name)
        print('Found site details file', sitelist_name)
    else:
        site_details = site_details_creator.site_details_creator(desired_lats, desired_lons, resource_dir)
        # Filter to locations in USA
        site_details = filter_sites(site_details, location='usa only')
        print('Site details: ', site_details)
        site_details.to_csv(sitelist_name)
# print("Site Details Created")

# case ='Case 1'
atb_year = 2022 #,2025,2030,2035]
ptc_options = 'yes'#, 'no']
sample_site['year'] = resource_year
useful_life = 25
critical_load_factor = 1
run_reopt_flag = False
custom_powercurve = True
storage_used = False
battery_can_grid_charge = False
grid_connected_hopp = False
interconnection_size_mw = 100
#electrolyzer_size = 50 #moved earlier in the code!

# which plots to show
plot_power_production = True
plot_battery = True
plot_grid = True
plot_h2 = True
plot_diesel=True
turbine_name =  'EWT_DW61-1000'
h2_model ='Simple'  

#Load information
scenario = dict()
df = pd.read_csv('/Users/cclark2/Desktop/HOPP/HOPP/examples/H2_Analysis/results_Nome/2021_2022_Energy Data_hourly.csv', skiprows=0, usecols = ['NJU-TOTAL-SYSTEM-LOAD (kW)', 'EWT1--3PHS-KW', 'EWT2--3PHS-KW'])
load = df['NJU-TOTAL-SYSTEM-LOAD (kW)']  #[kw_continuous for x in range(0, 8760)]  # * (sin(x) + pi) Set desired/required load profile for plant
load = load.astype(float).tolist()

#Existing turbine production data
EWT1_production = df['EWT1--3PHS-KW']
EWT1_production = EWT1_production.astype(float).tolist()
EWT2_production = df['EWT2--3PHS-KW']
EWT2_production = EWT2_production.astype(float).tolist()

#SAM Production Data
df_EWT = pd.read_csv('/Users/cclark2/Documents/HOPP_public/HOPP-1/1_EWT_2018_data.csv')
EWT3_production = df_EWT['System power generated | (kW)']
EWT3_production=EWT3_production.astype(float).tolist()

scenario_choice = 'Nome Analysis'
# site_selection = 'Site 1'
parent_path = os.path.abspath('')
results_dir = '/Users/cclark2/Documents/HOPP_public/HOPP-1/examples/H2_Analysis/Nome/results/'
# print('results_dir: ', results_dir)

itc_avail = 'no'
discount_rate = 0.089

forced_sizes = True
force_electrolyzer_cost = True
forced_wind_size = 0 
forced_solar_size = 0 #sample_site['no_solar'] = True for forced_solar_size = 0 (line 271 ish)
forced_storage_size_mw = 0
forced_storage_size_mwh = 0
# storage_size_mwh_options = [i * forced_storage_size_mw for i in range(1,11)]


# !NEW! Electrolyzer Info##
electrolyzer_model_parameters ={ 
    'Modify BOL Eff':False, #capability not built-in yet
    'BOL Eff [kWh/kg-H2]':[], #capability not built-in yet
    'Modify EOL Degradation Value':True, #for defining what efficiency drop means the end of life has been reached
    'EOL Rated Efficiency Drop':13} #13% drop from BOL efficiency means replace the stack
grid_connection_scenario ='off-grid'
h2_prod_capacity_required_kgphr=0 #used if grid-connected and want to make a specific amount of h2 at every hour of the year
electrolysis_scale = [] #unused right now
#electrolyzer_direct_cost_kw = 500 #only used if running 'optimize' as control strategy
pem_control_type = 'basic'
#useful_life = 30 #only used to give number of stack replacements over lifetime
use_degradation_penalty = True
electrolyzer_size_mw = 2 #must be an integer
number_of_electrolyzer_stacks = 1 #electrolyzer_size_mw must be divisible by this number
#^how many stacks you can control, so if electrolyzer_size_mw = 50 and number_of_electrolyzer_stacks = 5
#then each "stack" is 10 MW and the stacks can be turned on or off independent of other stacks
make_hydrogen = False
make_hydrogen_with = 'curtailed_renewable_power'#['renewable_power','curtailed_renewable_power','renewable_and_diesel']
kw_continuous = electrolyzer_size_mw * 1000


#Diesel gen capabilities
num_cat_gen = 5
num_wrs_gen = 1

#if both - it runs wrs first then cat, but it is easily modifiable in simple_generator_diesel.run_diesel()
use_battery_shortfall=True #if this is true, the shortfall is the shortfall after utilizing storage!
use_diesel=True #if this is true, then runs the diesel model during an outage!
generator_type='WRS' #options: 'CAT' or 'WRS' or 'Both'
forced_diesel_sizes=True #if false, uses as many generators as it takes to get to 0 shortfall

force_diesel_cat=False #only used if forced_diesel_sizes is True and generator type is 'CAT' or 'Both'
force_cat_size_kw= num_cat_gen * 2825 #16500 #num_cat_gen * 2825 #5650 #only used if generator type is 'CAT' or 'Both'
#NOTE: if you want to use a specific amount of CAT generators,
#then set force_cat_size_kw = num_cat_gen * 2825 and set force_diesel_cat to TRUE

force_diesel_warsilla=True #only used if forced_diesel_sizes is True and generator type is 'WRS' or 'Both'
force_wrs_size_kw= num_wrs_gen * 9390 #16000 #num_wrs_gen * 9390 #10000 #only used if generator type is 'WRS' or 'Both'
#NOTE: if you want to use a specific amount of WRS generators,
#then set force_wrs_size_kw = num_wrs_gen * 9390 and set force_diesel_warsilla to True
diesal_input={'forced_diesel_sizes':forced_diesel_sizes,'force_diesel_cat':force_diesel_cat,'force_diesel_wrs':force_diesel_warsilla,'force_cat_size_kw':force_cat_size_kw,'force_wrs_size_kw':force_wrs_size_kw}

sell_price = False
buy_price = False

# Define Turbine Characteristics based on user selected turbine.
if turbine_name == 'EWT_DW61-1000':
    #This is the smaller distributed wind option that matches Nome's existing turbines
    custom_powercurve_path = 'EWT_DW61-1000.csv'
    tower_height = 60 #resource was downloaded from 69m like what they have in Nome, but for PySam's sake, we  made it 60m
    rotor_diameter = 61
    turbine_rating_mw = 1
    wind_cost_kw = 1310 #ATB 2022 Conservative/Moderate/Advanced CAPEX for Wind Class II

    
print("Powercurve Path: ", custom_powercurve_path)
# site_details = site_details.set_index('site_nums')
save_outputs_dict = establish_save_output_dict()
save_all_runs = list()

# for forced_storage_size_mwh in storage_size_mwh_options:
# for ptc_avail in ptc_options:
#     for atb_year in atb_years:
for i, site_deet in enumerate(site_details.iterrows()):
    # if i == 0: continue
    # else:
    site_deet = site_deet[1]
    print(site_deet)
    lat = site_deet['Lat']
    lon = site_deet['Lon']
    sample_site['lat'] = lat
    sample_site['lon'] = lon
    sample_site['no_solar'] = True
    sample_site['no_wind'] = False
    # sample_site['n_timesteps'] = 8760
    # solar_resource_file = '/Users/cclark2/Desktop/HOPP/HOPP/resource_files/wind/{}_{}_-69m_psmv3_60_2018.csv'.format(lat, lon)
    wind_resource_file = '/Users/cclark2/Documents/HOPP_public/HOPP-1/resource_files/wind/{}_{}_windtoolkit_2018_60min_60m.srw'.format(lat, lon)
    site = SiteInfo(sample_site, wind_resource_file=wind_resource_file, hub_height=tower_height)
    site_name = (str(lat)+","+str(lon))

    if atb_year == 2022:
        forced_electrolyzer_cost = 400
        wind_cost_kw = 1462 #1310
        solar_cost_kw = 1105
        storage_cost_kw = 219
        storage_cost_kwh = 286
        debt_equity_split = 60
        wind_om_cost_kw = 44

    scenario['Useful Life'] = useful_life
    scenario['Debt Equity'] = debt_equity_split
    scenario['PTC Available'] = ptc_options
    scenario['ITC Available'] = itc_avail
    scenario['Discount Rate'] = discount_rate
    scenario['Tower Height'] = tower_height
    scenario['Powercurve File'] = custom_powercurve_path
   
    if forced_sizes:
        solar_size_mw = forced_solar_size
        wind_size_mw = forced_wind_size
        storage_size_mw = forced_storage_size_mw
        storage_size_mwh = forced_storage_size_mwh
        if storage_size_mw > 0:
            storage_hours = storage_size_mwh / storage_size_mw #0 when no storage included
        else: 
            storage_hours = 0
            print('Storage_hours = 0')
    # if forced_diesel_sizes:
    #     if force_diesel_cat:
    #         force_cat_size_kw=5000 #rated at 2825
    #     if force_diesel_warsilla:
    #         force_wrs_size_kw=1000 #rated at 9390

    if storage_size_mw > 0:
        technologies = {
                        'wind':
                            {'num_turbines': np.floor(wind_size_mw / turbine_rating_mw),
                                'turbine_rating_kw': turbine_rating_mw*1000,
                                'hub_height': tower_height,
                                'rotor_diameter': rotor_diameter},
                        # 'pv':
                        #     {'system_capacity_kw': solar_size_mw * 1000}, 
                        'battery': {
                            'system_capacity_kwh': storage_size_mwh * 1000,
                            'system_capacity_kw': storage_size_mw * 1000
                            }
                        }
    else:
                technologies = {
                        'wind':
                            {'num_turbines': np.floor(wind_size_mw / turbine_rating_mw),
                                'turbine_rating_kw': turbine_rating_mw*1000,
                                'hub_height': tower_height,
                                'rotor_diameter': rotor_diameter}#,
                       # 'pv':
                        #    {'system_capacity_kw': solar_size_mw * 1000}
                        }


    ## CASE 1: 2 MW Wind, rest diesel (Wartsilla) - no cost info b/c business as usual (assume EWT #1 performs like all turbs)

    ## CASE 2: 7 MW Wind, rest Diesel (10 MW Diesel) - this will include cost of turbines (+5) and fuel savings
    #TODO: If there is curtailed energy, send it to the electrolyzer

    #CASE 3: 7 MW Wind, 2MW/4MWh battery, rest Diesel (10 MW Diesel) - this will include cost of turbines (+5) and battery and fuel savings
    #TODO: If there is curtailed energy, send it to the battery, then any extra to electrolyzer

    existing_wind_production = tuple([i * wind_size_mw for i in EWT1_production]) #(kW) #df.iloc[:,1:].sum(axis=1)

    hybrid_plant, combined_pv_wind_power_production_hopp, combined_pv_wind_curtailment_hopp,\
        energy_shortfall_hopp, gen, annual_energies, wind_plus_solar_npv, npvs, lcoe =  \
        hopp_for_Nome(site, scenario, technologies, wind_size_mw, solar_size_mw, \
        storage_size_mw, storage_size_mwh, storage_hours, wind_cost_kw, solar_cost_kw, \
        storage_cost_kw, storage_cost_kwh, kw_continuous, load, custom_powercurve, \
        electrolyzer_size_mw, existing_wind_production, grid_connected_hopp=True, wind_om_cost_kw=wind_om_cost_kw)


    wind_installed_cost = hybrid_plant.wind.total_installed_cost
    if solar_size_mw > 0:
        solar_installed_cost = hybrid_plant.pv.total_installed_cost
    else:
        solar_installed_cost = 0
    hybrid_installed_cost = hybrid_plant.grid.total_installed_cost

    
    bat_model = SimpleDispatch()
    bat_model.Nt = len(energy_shortfall_hopp)
    bat_model.curtailment = combined_pv_wind_curtailment_hopp
    bat_model.shortfall = energy_shortfall_hopp
    bat_model.battery_storage = storage_size_mwh * 1000
    bat_model.charge_rate = storage_size_mw * 1000
    bat_model.discharge_rate = storage_size_mw * 1000
    battery_used, excess_energy, battery_SOC = bat_model.run()

    #Calculate Metrics after deploying battery

    combined_pv_wind_storage_power_production_hopp = combined_pv_wind_power_production_hopp + battery_used
    energy_shortfall_post_battery_hopp = [x - y for x, y in
                    zip(load,combined_pv_wind_storage_power_production_hopp)]
    energy_shortfall_post_battery_hopp = [x if x > 0 else 0 for x in energy_shortfall_post_battery_hopp]
    combined_pv_wind_curtailment_post_battery_hopp = [x - y for x, y in
                    zip(combined_pv_wind_storage_power_production_hopp,load)]
    combined_pv_wind_curtailment_post_battery_hopp = [x if x > 0 else 0 for x in combined_pv_wind_curtailment_post_battery_hopp]

    #Integrate simple diesel use model
    if use_battery_shortfall:
        diesel_shortfall=energy_shortfall_post_battery_hopp
    else:
        diesel_shortfall=energy_shortfall_hopp
    if use_diesel:
        diesal_input.update({'Shortfall':diesel_shortfall,'Generator Type(s)':generator_type})
        # diesal_input.update({'Shortfall':diesel_shortfall[892:899],'Generator Type(s)':generator_type})
        # diesal_input.update({'Shortfall':diesel_shortfall[3206:3254],'Generator Type(s)':generator_type})

        diesal_output={}
        diesel_model=SimpleDiesel(diesal_input,diesal_output)
        final_diesel_shortfall,total_diesel_power_supplied,total_diesel_fuel_used,total_diesel_cost=diesel_model.run_diesel()
        detailed_diesel_cost,detailed_diesel_performance,dietailed_diesel_specs=diesel_model.reorg_output_dict()
        
        # print(final_diesel_shortfall)
        # print(energy_shortfall_post_battery_hopp[3206:3254])
        
        df_load_percent = pd.DataFrame([energy_shortfall_hopp, energy_shortfall_post_battery_hopp, final_diesel_shortfall], ['Generation Shortfall', 'Gen + Battery Shortfall', 'Gen + Battery + Diesel Shortfall'])
        df_load_percent.to_csv(results_dir + 'Shortfall.csv')
        
        # combined_pv_wind_storage_power_production_hopp_outage = combined_pv_wind_storage_power_production_hopp[892:899]
        # combined_pv_wind_storage_power_production_hopp_outage = combined_pv_wind_storage_power_production_hopp[3206:3254]
        combined_pv_wind_storage_diesel_power_production_hopp = [total_diesel_power_supplied[i] + combined_pv_wind_storage_power_production_hopp[i] for i in range(len(total_diesel_power_supplied))]

        if plot_diesel:
            if generator_type=='Both':
                diesel_plot=['CAT','WRS']
            else:
                diesel_plot=[generator_type]
            diesel_model.post_process_diesel(diesel_plot,results_dir,saveme=True)
 
   #Check Percentage of time load is met with storage
    # import check_load_met
    if make_hydrogen:
            #['renewable_power','curtailed_renewable_power','renewable_and_diesel']
            if make_hydrogen_with=='curtailed_renewable_power':
                power_to_electrolyzer = np.array(combined_pv_wind_curtailment_post_battery_hopp)
            elif make_hydrogen_with=='renewable_power':
                power_to_electrolyzer = np.array(combined_pv_wind_storage_power_production_hopp)
            elif make_hydrogen_with =='renewable_and_diesel' and use_diesel:
                power_to_electrolyzer = np.array(total_diesel_power_supplied) + np.array(combined_pv_wind_storage_power_production_hopp)
           
            H2_Results,H2_Ts_Data,H2_Agg_data,energy_signal_to_electrolyzer = run_h2_PEM.run_h2_PEM(power_to_electrolyzer, electrolyzer_size_mw,
            useful_life, number_of_electrolyzer_stacks,  electrolysis_scale, pem_control_type, forced_electrolyzer_cost ,electrolyzer_model_parameters, 
            use_degradation_penalty,grid_connection_scenario,h2_prod_capacity_required_kgphr)

            plot_h2 = True
            plot_results.plot_h2_results(H2_Results, energy_signal_to_electrolyzer,\
                results_dir,scenario_choice,atb_year,load,plot_h2)


    #Generate 50, 75, 90, 95% load profiles
    load_percentages = [0.5,0.75,0.9,0.95,1]
    for load_perc in load_percentages:
        load_at_perc = [i * load_perc for i in load]
        globals()[f"perc_time_load_met_no_storage_{int(100*load_perc)}"],globals()[f"perc_time_load_met_with_storage_{int(100*load_perc)}"],\
        globals()[f"energy_shortfall_no_storage_{int(100*load_perc)}"],globals()[f"energy_shortfall_with_storage_{int(100*load_perc)}"] = \
            check_load_met.check_load_met(load_at_perc,combined_pv_wind_power_production_hopp,combined_pv_wind_storage_power_production_hopp) 
            # check_load_met.check_load_met(load_at_perc,combined_pv_wind_power_production_hopp,combined_pv_wind_storage_power_production_hopp) 

    #Check Percentage of time load is met with diesel
     # import check_load_met
     #Generate 50, 75, 90, 95% load profiles
    load_percentages = [0.5,0.75,0.9,0.95,1]
    for load_perc in load_percentages:
         load_at_perc = [i * load_perc for i in load]
         globals()[f"perc_time_load_met_no_diesel_{int(100*load_perc)}"],globals()[f"perc_time_load_met_with_diesel_{int(100*load_perc)}"],\
         globals()[f"energy_shortfall_no_diesel_{int(100*load_perc)}"],globals()[f"energy_shortfall_with_diesel_{int(100*load_perc)}"] = \
             check_load_met.check_load_met(load_at_perc,combined_pv_wind_storage_power_production_hopp,combined_pv_wind_storage_diesel_power_production_hopp) 
             # check_load_met.check_load_met(load_at_perc,combined_pv_wind_power_production_hopp,combined_pv_wind_storage_power_production_hopp) 


    #Generate text for plots
    battery_plot_text = '{}MW, {}MWh battery'.format(storage_size_mw,storage_size_mwh)
    battery_plot_results_text = 'Shortfall Before: {0:,.0f}\nShortfall After: {1:,.0f}'.format(
        np.sum(energy_shortfall_hopp),np.sum(energy_shortfall_post_battery_hopp))
    percentage_time_load_met_results_text = 'Without Storage, this configuration meets:\n50% load {}% of the time \n75% load {}% of the time\n90% load {}% of the time\n95% load {}% of the time and\n100% load {}% of the time'.format(perc_time_load_met_no_storage_50, perc_time_load_met_no_storage_75,perc_time_load_met_no_storage_90,perc_time_load_met_no_storage_95,perc_time_load_met_no_storage_100)
    percentage_time_load_met_with_storage_results_text = 'With {}MW and {}MWh Storage, this configuration meets:\n50% load {}% of the time \n75% load {}% of the time\n90% load {}% of the time\n95% load {}% of the time and\n100% load {}% of the time'.format(storage_size_mw, storage_size_mwh, perc_time_load_met_with_storage_50, perc_time_load_met_with_storage_75,perc_time_load_met_with_storage_90,perc_time_load_met_with_storage_95,perc_time_load_met_with_storage_100)
    print('==============================================')
    print(battery_plot_results_text)
    print(percentage_time_load_met_results_text)
    print(percentage_time_load_met_with_storage_results_text)
    print('==============================================')
    

    
    if plot_battery:
        
        plt.figure(figsize=(9,6))
        plt.subplot(311)
        plt.plot(combined_pv_wind_curtailment_hopp[3206:3254],label="Curtailment") #outage from 892-898
        plt.plot(final_diesel_shortfall[3206:3254],"--",label="Shortfall")
        plt.ylabel('Curtailment (kW) and \n Shortfall (kW)')
        plt.xlabel('Time (hours)')
        # plt.plot(energy_shortfall_hopp[3206:3254],label="Shortfall")
        # plt.title(battery_plot_text + '\n' + battery_plot_results_text + '\n' + 'Energy Curtailment and Shortfall')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.subplot(312)
        plt.plot(combined_pv_wind_storage_diesel_power_production_hopp[3206:3254],label="Wind+Storage+Diesel") #label="Wind+PV+Storage+Diesel")
        plt.plot(combined_pv_wind_storage_power_production_hopp[3206:3254],label="Wind+Storage")
        plt.plot(combined_pv_wind_power_production_hopp[3206:3254],"--",label="Wind")
        # plt.plot(wind_power_production_hopp[3206:3254],"--",label="Wind")
        # plt.plot(pv_power_production_hopp[3206:3254],"--",label="PV")
        plt.plot(load[3206:3254],"--",label="Load")
        plt.ylabel('Hybrid Plant Component \n Contributions (kW)')
        plt.xlabel('Time (hours)')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.title("Hybrid Plant Power Energy Contributions")
        # plt.title("Hybrid Plant Power Flows with and without Storage")
        plt.tight_layout()

        plt.subplot(313)
        plt.plot(battery_SOC[3206:3254],label="State of Charge")
        plt.plot(battery_used[3206:3254],"--",label="Battery Used")
        plt.ylabel('Battery State of Charge (kW) \n and Capacity Used (kW)')
        plt.xlabel('Time (hours)')
        # plt.title('Battery State')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(os.path.join(results_dir,'May Resilience Battery Sizing_{}MW_{}MWh.jpg'.format(storage_size_mw,storage_size_mwh)),bbox_inches='tight', format="png", dpi=1200)
        # plt.show()



        plt.figure(figsize=(9,6))
        plt.subplot(311)
        plt.plot(combined_pv_wind_curtailment_hopp[745:1416],label="Curtailment") #outage from 892-898
        plt.plot(final_diesel_shortfall[745:1416],"--",label="Shortfall")
        plt.ylabel('Curtailment (kW) and \n Shortfall (kW)')
        plt.xlabel('Time (hours)')
        # plt.plot(energy_shortfall_hopp[3206:3254],label="Shortfall")
        # plt.title(battery_plot_text + '\n' + battery_plot_results_text + '\n' + 'Energy Curtailment and Shortfall')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.subplot(312)
        plt.plot(combined_pv_wind_storage_diesel_power_production_hopp[745:1416],label="Wind+Storage+Diesel") #label="Wind+PV+Storage+Diesel")
        plt.plot(combined_pv_wind_storage_power_production_hopp[745:1416],label="Wind+Storage")
        plt.plot(combined_pv_wind_power_production_hopp[745:1416],"--",label="Wind")
        # plt.plot(wind_power_production_hopp[3206:3254],"--",label="Wind")
        # plt.plot(pv_power_production_hopp[3206:3254],"--",label="PV")
        plt.plot(load[745:1416],"--",label="Load")
        plt.ylabel('Hybrid Plant Component \n Contributions (kW)')
        plt.xlabel('Time (hours)')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.title("Hybrid Plant Power Energy Contributions")
        # plt.title("Hybrid Plant Power Flows with and without Storage")
        plt.tight_layout()

        plt.subplot(313)
        plt.plot(battery_SOC[745:1416],label="State of Charge")
        plt.plot(battery_used[745:1416],"--",label="Battery Used")
        plt.ylabel('Battery State of Charge (kW) \n and Capacity Used (kW)')
        plt.xlabel('Time (hours)')
        # plt.title('Battery State')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(os.path.join(results_dir,'Feb Resilience Battery Sizing_{}MW_{}MWh.jpg'.format(storage_size_mw,storage_size_mwh)),bbox_inches='tight', format="png", dpi=1200)
        # plt.show()




        plt.figure(figsize=(9,6))
        plt.subplot(311)
        plt.plot(combined_pv_wind_curtailment_hopp,label="Curtailment") #outage from 892-898
        plt.plot(final_diesel_shortfall,"--",label="Shortfall")
        plt.ylabel('Curtailment (kW) and \n Shortfall (kW)')
        plt.xlabel('Time (hours)')
        # plt.plot(energy_shortfall_hopp[3206:3254],label="Shortfall")
        # plt.title(battery_plot_text + '\n' + battery_plot_results_text + '\n' + 'Energy Curtailment and Shortfall')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.subplot(312)
        plt.plot(combined_pv_wind_storage_diesel_power_production_hopp,label="Wind+Storage+Diesel") #label="Wind+PV+Storage+Diesel")
        plt.plot(combined_pv_wind_storage_power_production_hopp,label="Wind+Storage")
        plt.plot(combined_pv_wind_power_production_hopp,"--",label="Wind")
        # plt.plot(wind_power_production_hopp[3206:3254],"--",label="Wind")
        # plt.plot(pv_power_production_hopp[3206:3254],"--",label="PV")
        plt.plot(load,"--",label="Load")
        plt.ylabel('Hybrid Plant Component \n Contributions (kW)')
        plt.xlabel('Time (hours)')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.title("Hybrid Plant Power Energy Contributions")
        # plt.title("Hybrid Plant Power Flows with and without Storage")
        plt.tight_layout()


        plt.subplot(313)
        plt.plot(battery_SOC,label="State of Charge")
        plt.plot(battery_used,"--",label="Battery Used")
        plt.ylabel('Battery State of Charge (kW) \n and Capacity Used (kW)')
        plt.xlabel('Time (hours)')
        # plt.title('Battery State')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(os.path.join(results_dir,'ALL Resilience Battery Sizing_{}MW_{}MWh.jpg'.format(storage_size_mw,storage_size_mwh)),bbox_inches='tight', format="png", dpi=1200)


        
        print('total diesel cost: ', total_diesel_cost)
        print('total shortfall: ', sum(final_diesel_shortfall))

        df_outage = pd.DataFrame({'Curtailment': combined_pv_wind_curtailment_hopp, 
                                  'Shortfall': energy_shortfall_hopp, 
                                  'Wind + PV + Storage': combined_pv_wind_storage_power_production_hopp, 
                                  'Wind + PV Generation': combined_pv_wind_power_production_hopp, 
                                  'Load': load,'State of Charge': battery_SOC, 'Battery Used': battery_used,
                                  'Final Diesel Shortfall': final_diesel_shortfall,
                                  'Total Diesel Power Supplied': total_diesel_power_supplied,
                                  'Total Diesel Fuel Used': total_diesel_fuel_used})
        df_outage.to_csv(os.path.join(results_dir, "DIESEL_Hybrid_TotalLoad_Nome_{}_{}.csv".format(scenario_choice, forced_storage_size_mwh)))



    if plot_grid:
        plt.plot(combined_pv_wind_storage_power_production_hopp[200:300],label="before buy from grid")
        plt.suptitle("Power Signal Before Purchasing From Grid")
    
    if sell_price:
        profit_from_selling_to_grid = np.sum(excess_energy)*sell_price
    else:
        profit_from_selling_to_grid = 0.0

    # buy_price = False # if you want to force no buy from grid
    if buy_price:
        cost_to_buy_from_grid = buy_price

        for i in range(len(combined_pv_wind_storage_power_production_hopp)):
            if combined_pv_wind_storage_power_production_hopp[i] < kw_continuous:
                cost_to_buy_from_grid += (kw_continuous-combined_pv_wind_storage_power_production_hopp[i])*buy_price
                combined_pv_wind_storage_power_production_hopp[i] = kw_continuous
    else:
        cost_to_buy_from_grid = 0.0

    energy_to_electrolyzer = [x if x < kw_continuous else kw_continuous for x in combined_pv_wind_storage_power_production_hopp]

    electrical_generation_timeseries = np.zeros_like(energy_to_electrolyzer)
    electrical_generation_timeseries[:] = energy_to_electrolyzer[:]

    adjusted_installed_cost = hybrid_plant.grid._financial_model.Outputs.adjusted_installed_cost
    #NB: adjusted_installed_cost does NOT include the electrolyzer cost
    useful_life = scenario['Useful Life']
    net_capital_costs = 0

    # system_rating = electrolyzer_size
    system_rating = wind_size_mw + solar_size_mw
    # Power = [(est_const_desal_power_mw_hr) * 1000 for x in range(0, 8760)]
    Power = electrical_generation_timeseries

    total_elec_production = np.sum(electrical_generation_timeseries)
    total_hopp_installed_cost = hybrid_plant.grid._financial_model.SystemCosts.total_installed_cost

    total_system_installed_cost = total_hopp_installed_cost 

    total_annual_operating_costs = cost_to_buy_from_grid - profit_from_selling_to_grid

    # Cashflow Financial Calculation
    discount_rate = scenario['Discount Rate']
    cf_wind_annuals = hybrid_plant.wind._financial_model.Outputs.cf_annual_costs
    if solar_size_mw > 0:
        cf_solar_annuals = hybrid_plant.pv._financial_model.Outputs.cf_annual_costs
    else:
        cf_solar_annuals = np.zeros(useful_life)

    cf_operational_annuals = [-total_annual_operating_costs for i in range(useful_life)]

    cf_df = pd.DataFrame([cf_wind_annuals, cf_solar_annuals],['Wind', 'Solar'])

    cf_df.to_csv(os.path.join(results_dir, "Annual Cashflows_{}_{}_{}_discount_{}.csv".format(site_name, scenario_choice, atb_year, discount_rate)))

    #NPVs of wind, solar, H2
    npv_wind_costs = npf.npv(discount_rate, cf_wind_annuals)
    npv_solar_costs = npf.npv(discount_rate, cf_solar_annuals)

    npv_operating_costs = npf.npv(discount_rate, cf_operational_annuals)


    npv_total_costs = npv_wind_costs+npv_solar_costs
    npv_total_costs_w_operating_costs = npv_wind_costs+npv_solar_costs+npv_operating_costs

    print_results = False
    print_h2_results = True
    save_outputs_dict = save_the_things_alt()
    save_all_runs.append(save_outputs_dict)
    save_outputs_dict = establish_save_output_dict()


print_results = False
if print_results:
    # ------------------------- #
    #TODO: Tidy up these print statements
    print("Future Scenario: {}".format(scenario['Scenario Name']))
    print("Wind Cost per KW: {}".format(scenario['Wind Cost KW']))
    print("PV Cost per KW: {}".format(scenario['Solar Cost KW']))
    print("Storage Cost per KW: {}".format(scenario['Storage Cost kW']))
    print("Storage Cost per KWh: {}".format(scenario['Storage Cost kWh']))
    print("Wind Size built: {}".format(wind_size_mw))
    print("PV Size built: {}".format(solar_size_mw))
    print("Storage Size built: {}".format(storage_size_mw))
    print("Storage Size built: {}".format(storage_size_mwh))
    print("Levelized cost of Electricity (HOPP): {}".format(lcoe))
    print("Total Yearly Electrical Output: {}".format(total_elec_production))
    # ------------------------- #
save_esg=True
if save_esg:
    # gen_data={'wind_gen':hybrid_plant.generation_profile.wind,'solar_gen':hybrid_plant.generation_profile.pv,'battery_gen':hybrid_plant.generation_profile.battery}
    # save_df=pd.DataFrame(gen_data)
    df_outage = pd.DataFrame([combined_pv_wind_curtailment_hopp, 
                                              energy_shortfall_hopp, 
                                              combined_pv_wind_storage_power_production_hopp, 
                                              combined_pv_wind_power_production_hopp, load,
                                              battery_SOC, battery_used,],
                                              ['Curtailment', 'Shortfall', 'Wind + PV + Storage', 
                                              'Wind + PV Generation', 'Load', 'State of Charge', 'Battery Used'])
    #df_outage.to_csv('/Users/egrant/Desktop/MRCL_Things/HOPP-Algona/MRCL-ESG/RESULTS-11-29/2013_algona_gen_data.csv')
    df_outage.to_csv(results_dir + 'ESG_Diesel_Tests_Y{}_wind{}_solar{}_storage{}_algona_gen_data.csv'.format(resource_year, forced_wind_size, forced_solar_size, forced_storage_size_mwh))
    #df_outage.to_csv(results_dir + 'ESG_Y{}_wind{}_solar{}_storage{}_algona_gen_data.csv'.format(resource_year, forced_wind_size, forced_solar_size, forced_storage_size_mwh))
save_outputs = True #ESG
if save_outputs:
    #save_outputs_dict_df = pd.DataFrame(save_all_runs)
    save_all_runs_df = pd.DataFrame(save_all_runs)
    #save_all_runs_df.to_csv(os.path.join(results_dir, "ESG_Feb_ResilienceAnalysisMIRACL_Algona_VestasV82.csv"))
    save_all_runs_df.to_csv(results_dir + 'ResilienceAnalysis_VestasV82_Y{}_wind{}_solar{}_storage{}.csv'.format(resource_year, forced_wind_size, forced_solar_size, forced_storage_size_mwh))


