# implement a simple dispatch model
import math
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

class SimpleDiesel():

    def __init__(self,input_dict,output_dict):
        self.input_dict=input_dict
        self.output_dict=output_dict

        self.gen_type=self.input_dict['Generator Type(s)']
        self.cost_of_diesel_fuel=4.39 #[$/gal]

        if self.input_dict['forced_diesel_sizes']:
            self.forcing_sizes()
        else:
            self.num_cats=-1
            self.num_wrs=-1

        #fuel_consump=188.8 #gal/hr specified at 2825 kW (prime)
        # heat_rate=9223 #BTU/kWH
        # self.Nt=1 #number of hours in simulation
        # self.shortfall = np.zeros(self.Nt)
         #[$/gal] based on 01/09/2023 data for the widwest
        # ref_power_kw=2825 #kW (prime)
        # genset_rating_kW=3100 #standby = v_line*i_rated*pf*sqrt(3)
        # genset_rating_kVa=3875
        # power_factor= genset_rating_kW/genset_rating_kVa#0.8
        # v_line=13800
        # v_phase=7967
        # i_rated=162.1
        # freq=60 #Hz
        # gen_eff=0.37
        # field_current=[0,20.7,24.2,27.6,31.1,34.5,38.0,41.5,44.9,48.4]
        # l2l_voltage=[0,8280,9660,11040,12420,13800,15180,16560,17940,19320]
        # load_x=genset_rating_kW*[0.25,0.5,0.75,1] #kW
        # fuel_consump_y=[66.3,109.7,154.6,212.9] #gal/hr
        # fuel_per_load=np.interp(power_demand,load_x,fuel_consump_y)
        # fuel_temp_C=29
        # fuel_lhv=42780 #kJ/kg
        # fuel_weight=838.9 #g/liter

        #200 hrs/year operational, max of 500 hours/year
        #average power output is 85% of standby power rating
        #peak demand upto 100% of standby rated for 5% of the operating time

        #CONNECTION IS SERIES STAR
        #1CF=1026 BTW
        # btu_per_gal=138000
        # btw_per_cf=1026
        #1 gal = 138000 BTU
    def forcing_sizes(self):
        '''This function is called if forced_diesel_sizes is true
        -rounds the number of generators to the nearest number based on the generator's installed capacity
        -if the number of generators (num_cats or num_wrs) =-1 then the generator model will run 
        as many generators as it takes to get to zero shortfall'''
        if self.input_dict['force_diesel_cat']:
            self.caterpillar_gen()
            self.num_cats=round(self.input_dict['force_cat_size_kw']/self.output_dict['CAT-Specs']['Rated Capacity'])
        else:
            self.num_cats=-1

        if self.input_dict['force_diesel_wrs']:
            self.wartsila_diesel_gen()
            self.num_wrs=round(self.input_dict['force_wrs_size_kw']/self.output_dict['WRS-Specs']['Rated Capacity'])
        else:
            self.num_wrs=-1

    def caterpillar_gen(self):
        '''This function initializes the specs for the caterpillar diesel generator model'''
        rated_capacity=2825 #kW
        genset_rating_kW=3100
        genset_rating_kVa=3875
        power_factor= genset_rating_kW/genset_rating_kVa
        gen_eff=0.37
        capex_cost=1750000 #[$/genset]
        load_x=list(genset_rating_kW*np.array([0.25,0.5,0.75,1])) #kW
        fuel_consump_y=[66.3,109.7,154.6,212.9] #gal/hr
        #fuel_per_load=np.interp(power_demand,load_x,fuel_consump_y)
        self.output_dict['CAT-Specs']={'Rated Capacity':rated_capacity,'Maximum Output':genset_rating_kW,'Gen-Eff':gen_eff,\
            'CapEx':capex_cost,'Load':load_x,'Fuel Consump':fuel_consump_y}

    def wartsila_diesel_gen(self):
        '''This function initializes the specs for the Wartsila diesel generator model'''
        #https://www.wartsila.com/docs/default-source/product-files/engines/df-engine/w34df-product-guide.pdf?sfvrsn=5116cb45_23
        #wartsilla 16V34DF Genset with 750 rpm (50Hz) enginer is 8000kW and generator is 9600 kVA
        #50-70 secon start-up
        rated_capacity=9390 #kW - max enginer output is 110% of rated
        gen_eff=0.41
        fuel_consump=567 #gal/hr
        capex_cost=6200000 # [$/gen_set]
        self.output_dict['WRS-Specs']={'Rated Capacity':rated_capacity,'Maximum Output':1.1*rated_capacity,'Gen-Eff':gen_eff,\
            'CapEx':capex_cost,'Load':rated_capacity,'Fuel Consump':fuel_consump}
        #paper has g/kWh
        # length of simulation

    def run_diesel(self):
        '''this function is what runs the diesel models depending on the gen_type
        -it also calls the post-processing type function to get the main outputs of interest!'''
        #determine how many or which type of generators to use
        
        if self.gen_type=='CAT':
            self.caterpillar_gen()
            self.run_caterpillar(num_gen=self.num_cats)
        elif self.gen_type=='WRS':
            self.wartsila_diesel_gen()
            self.run_warsilla(num_gen=self.num_wrs)
        elif self.gen_type=='Both':
            self.caterpillar_gen()
            self.wartsila_diesel_gen()
            self.run_warsilla(num_gen=self.num_wrs)
            self.run_caterpillar(num_gen=self.num_cats)
        self.simple_cost()
        cost,performance,specs=self.reorg_output_dict()
        tot_sf,tot_power,tot_fuel,tot_cost=self.give_final_numbers(performance,cost)
        return tot_sf,tot_power,tot_fuel,tot_cost#cost,performance,specs
            #need to pass shortfall too!
            #self.run_caterpillar(set_gen_num)
    def reorg_output_dict(self):
        '''this function reorganizes the huge output dictionary
        into smaller dictionaries
        
        - cost_dict has the capex, fuel cost, and total cost for each generator type
        and each individual generator of that type
        -spec_dict has the specifications for each generator type used in the code
        -perf_dict has the cumulative shortfall for each generator type and individual generator,
        has the power supplied for each individual generator, and has the fuel consumption for each
        individual generator'''
        keys,vals=zip(*self.output_dict.items())
        cost_dict={}
        spec_dict={}
        perf_dict={}
        for idx,key in enumerate(keys):
            if 'Cost' in key:
                cost_dict.update({key:vals[idx]})
            if 'Specs' in key:
                spec_dict.update({key:vals[idx]})
            if 'Performance' in key:
                perf_dict.update({key:vals[idx]})

        return cost_dict,perf_dict,spec_dict
    
    def give_final_numbers(self,perf_dict,cost_dict):
        '''This outputs final shortfall, total power supplied by the generators,
        total fuel consumed by the generators and total cost (CapEx + Fuel consumption)'''
        cost_keys,cost_vals=zip(*cost_dict.items())
        gen_type_keys,vals=zip(*perf_dict.items())
        final_sf=np.zeros(len(self.input_dict['Shortfall']))
        final_power_supplied=np.zeros(len(self.input_dict['Shortfall']))
        final_fuel_consump=np.zeros(len(self.input_dict['Shortfall']))
        total_cost=0
        for gi,gen in enumerate(gen_type_keys):
            num_gen_key,num_gen_dict =zip(*perf_dict[gen]['Individual Generator'].items())
            final_sf+= perf_dict[gen]['Individual Generator'][num_gen_key[-1]]['Total Shortfall']
            cost_df=pd.DataFrame(cost_dict[cost_keys[gi]],index=num_gen_key)
            for gen_num in num_gen_key:
                final_power_supplied +=perf_dict[gen]['Individual Generator'][gen_num]['Power Supplied (1 Gen)']
                final_fuel_consump +=perf_dict[gen]['Individual Generator'][gen_num]['Fuel Consumption (1 Gen)']
                total_cost += cost_df.loc[gen_num]['Total Cost']

        return final_sf, final_power_supplied, final_fuel_consump, total_cost


    def simple_cost(self):
        '''Simple cost function based on total fuel consumption & CapEx'''
        if self.gen_type=='Both':
            gen_names=['CAT','WRS']
        else:
            gen_names=[self.gen_type]
        for gen_type in gen_names:
            capex_1gen_dollar=[self.output_dict[gen_type + '-Specs']['CapEx']]
            perf_dict=self.output_dict[gen_type + '-Performance']['Individual Generator']
            fuel_consump_gals=[np.sum(perf_dict[gen_i]['Fuel Consumption (1 Gen)']) for gen_i in list(perf_dict.keys())]
            fuel_cost_per_gen_dollars=self.cost_of_diesel_fuel*np.array(fuel_consump_gals)
            gen_nums=list(perf_dict.keys())
            self.output_dict[gen_type + '-Cost [$]']=dict(zip(['Gen Number','CapEx','Fuel Cost','Total Cost'],[gen_nums,capex_1gen_dollar*len(gen_nums),fuel_cost_per_gen_dollars,fuel_cost_per_gen_dollars+capex_1gen_dollar,capex_1gen_dollar*len(gen_nums) + fuel_cost_per_gen_dollars]))

    def run_caterpillar(self,num_gen=-1):
        '''this function runs the caterpillar generator'''
        #inputs are: total shortfall
        #number of generators
        #generator type
        
        run_dict=self.output_dict['CAT-Specs']
        init_sf=self.input_dict['Shortfall']
        if self.gen_type=='Both':
            if 'Individual Generator' in list(self.output_dict['WRS-Performance'].keys()):
                other_gen_key=list(self.output_dict['WRS-Performance']['Individual Generator'].keys())[-1]
                init_sf=self.output_dict['WRS-Performance']['Individual Generator'][other_gen_key]['Total Shortfall']
        if num_gen<0:
            
            num_generators=np.ceil(np.max(init_sf)/run_dict['Rated Capacity']) + 1
        else:
            num_generators=num_gen
        
        multi_gen_dict={}
        self.Nt=len(init_sf)
        new_shortfall=np.copy(init_sf)
        
        for ngen in np.arange(0,round(num_generators),1):
            multi_gen_dict['Gen #' + str(ngen)]={}
            tot_sf=[]
            tot_power=[]
            fuel_consumed=[]
            for i in range(self.Nt):
                #if init_sf[i]<self.output_dict['Rated Capacity']:
                power_supplied=new_shortfall[i]
                #if power_supplied>run_dict['Rated Capacity'] & power_supplied < run_dict['Maximum Output']

                if power_supplied>run_dict['Rated Capacity']:
                    power_supplied=run_dict['Rated Capacity']
                    new_shortfall[i]=new_shortfall[i]-run_dict['Rated Capacity']
                    
                else:
                    new_shortfall[i]=0
                
                if power_supplied>0:
                    fuel_per_load=np.interp(power_supplied,run_dict['Load'],run_dict['Fuel Consump'])
                else:
                    fuel_per_load=0
                    
                fuel_consumed.append(fuel_per_load)
                tot_sf.append(new_shortfall[i])
                #tot_sf.append(new_shortfall[i])
                tot_power.append(power_supplied)
            multi_gen_dict['Gen #' + str(ngen)]={'Total Shortfall': tot_sf,'Power Supplied (1 Gen)': tot_power,'Fuel Consumption (1 Gen)': fuel_consumed}
            df_diesel = pd.DataFrame.from_dict(multi_gen_dict)
            df_diesel.to_csv('Diesel_Outputs.csv')
            
            
        self.output_dict['CAT-Performance']={'Individual Generator':multi_gen_dict}

    def run_warsilla(self,num_gen=-1):
        '''This function runs the warsilla generator'''
        #inputs are: total shortfall
        #number of generators
        #generator type
        run_dict=self.output_dict['WRS-Specs']
        init_sf=self.input_dict['Shortfall']
        if self.gen_type=='Both':
            if 'CAT-Performance' in list(self.output_dict.keys()):
                if 'Individual Generator' in list(self.output_dict['CAT-Performance'].keys()):
                    other_gen_key=list(self.output_dict['CAT-Performance']['Individual Generator'].keys())[-1]
                    init_sf=self.output_dict['CAT-Performance']['Individual Generator'][other_gen_key]['Total Shortfall']
        if num_gen <0:
            num_generators=np.ceil(np.max(init_sf)/run_dict['Rated Capacity']) + 1
        else:
            num_generators=num_gen
        
        multi_gen_dict={}
        self.Nt=len(init_sf)
        new_shortfall=np.copy(init_sf)
        
        for ngen in np.arange(0,round(num_generators),1):
            multi_gen_dict['Gen #' + str(ngen)]={}
            tot_sf=[]
            tot_power=[]
            fuel_consumed=[]
            for i in range(self.Nt):
                #if init_sf[i]<self.output_dict['Rated Capacity']:
                power_supplied=new_shortfall[i]
                #if power_supplied>run_dict['Rated Capacity'] & power_supplied < run_dict['Maximum Output']

                if power_supplied>run_dict['Rated Capacity']:
                    power_supplied=run_dict['Rated Capacity']
                    new_shortfall[i]=new_shortfall[i]-run_dict['Rated Capacity']
                    
                else:
                    new_shortfall[i]=0
                fuel_per_load=power_supplied*(run_dict['Fuel Consump']/run_dict['Load']) #double check
                fuel_consumed.append(fuel_per_load)
                tot_sf.append(new_shortfall[i])
                #tot_sf.append(new_shortfall[i])
                tot_power.append(power_supplied)
            multi_gen_dict['Gen #' + str(ngen)]={'Total Shortfall': tot_sf,'Power Supplied (1 Gen)': tot_power,'Fuel Consumption (1 Gen)': fuel_consumed}
        
        self.output_dict['WRS-Performance']={'Individual Generator':multi_gen_dict}
        
    def post_process_diesel(self,generator_names,savepath,saveme=False):
        '''I am not sure whether this is totally functional at the moment but could be worth a try!'''
        cost,performance,specs=self.reorg_output_dict()
        keys,vals=zip(*performance.items())
        allgen_dict={}
        pergen_dict={}
        #gen_key=[[key,idx] for idx,key in enumerate(list(keys)) if generator in key]
        for generator in generator_names:
            
            #perf_key=[key for key in keys if generator in key]
            mini_dict=performance[generator + '-Performance']['Individual Generator'] 
            mini_cost=cost[generator + '-Cost [$]']
            #mini_dict=performance[perf_key[idx]]['Individual Generator'] #TODO - put in loop
            gen_keys=list(mini_dict.keys())
            # print(gen_keys)
            pergen_dict[generator]={}
            cumulative_power_prod=np.zeros((len(gen_keys)+1,len(self.input_dict['Shortfall'])))
            cumulative_fuel_consump=np.zeros((len(gen_keys)+1,len(self.input_dict['Shortfall'])))
            for gi,gen in enumerate(gen_keys):
                cumulative_power_prod[gi+1]=cumulative_power_prod[gi] + np.array(mini_dict[gen]['Power Supplied (1 Gen)'])
                cumulative_fuel_consump[gi+1]=cumulative_fuel_consump[gi] + np.array(mini_dict[gen]['Fuel Consumption (1 Gen)'])
                pergen_dict[generator].update({gen:{'Power Prod':list(cumulative_power_prod[gi+1]),'Fuel Consump':list(cumulative_fuel_consump[gi+1])}})
            allgen_power_prod=cumulative_power_prod[-1]
            allgen_fuel_consumed=cumulative_fuel_consump[-1]
            allgen_sf=mini_dict[gen_keys[-1]]['Total Shortfall']
            allgen_dict[generator]={'Power Produced [kWh]':allgen_power_prod,'Fuel Consumed [gal]':allgen_fuel_consumed,'Shortfall [kWh]':allgen_sf}

            plt.figure(figsize=(9,6))
            plt.subplot(311)
            plt.plot(self.input_dict['Shortfall'],color='black',label='No Generators')

            for gen in gen_keys:
                # print('Gen: ', gen)
                # print('Total Shortfall: ', mini_dict[gen]['Total Shortfall'])
                # df_diesel = pd.DataFrame([gen, mini_dict[gen]['Total Shortfall']], ['Gen', 'Total Shortfall'])
                # df_diesel.to_csv(savepath + 'Diesel_Outputs_shortfall.csv')
                
                plt.plot(mini_dict[gen]['Total Shortfall'],"--",label= gen)
                plt.ylabel('Shortfall [kWh]')
                plt.xlabel('Time (hours)')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #location='upper right')
            plt.subplot(312)
            for gi in np.arange(1,len(cumulative_power_prod)):
                # print('Gi: ', gi)
                # print('Diesel Energy Production: ', cumulative_power_prod[gi])
                # df_diesel = pd.DataFrame([gen, cumulative_power_prod[gi]], ['Gen', 'Diesel Energy \n Production [kW]'])
                # df_diesel.to_csv(savepath + 'Diesel_Outputs_power_gen.csv')

                plt.plot(cumulative_power_prod[gi],"--",label='+ {} Generators'.format(gi))
                plt.xlabel('Time (hours)')
                plt.ylabel('Diesel Energy \n Production [kW]')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                #plt.plot(mini_dict[gen]['Power Supplied (1 Gen)'],label= gen)
            plt.subplot(313)
            for gi in np.arange(1,len(cumulative_fuel_consump)):
                # print('Gi: ', gi)
                # print('Diesel Consumption: ', cumulative_fuel_consump[gi])
                # df_diesel = pd.DataFrame([gen, cumulative_fuel_consump[gi]], ['Gen', 'Diesel Consumption [gal]'])
                # df_diesel.to_csv(savepath + 'Diesel_Outputs_consumption.csv')

                plt.plot(cumulative_fuel_consump[gi],"--",label='+ {} Generators'.format(gi))
                # plt.set_ylim(ymin=0)
                plt.xlabel('Time (hours)')
                plt.ylabel('Diesel Fuel \n Consumption [gal]')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.suptitle('Cumulative Metrics for {} '.format(len(gen_keys)) + ' ' +generator + 'Generator(s)')
            plt.tight_layout()
            []
            if saveme:
                today=list(time.localtime())
                subdesc='-M{}_D{}_Y{}'.format(today[1],today[2],today[0])
                plt.savefig(savepath+'DIESEL_{}_{}_{}'.format(len(gen_keys), generator, subdesc),bbox_inches='tight', format="png", dpi=1200)
                #plt.plot(mini_dict[gen]['Fuel Consumption (1 Gen)'],label= gen)

            # df_outage = pd.DataFrame([mini_dict['Total Shortfall'], 
            #                               cumulative_power_prod, 
            #                               cumulative_fuel_consump], 
            #                               ['Shortfall (kWh)', 'Cumulative Diesel Power Production (kW)', 
            #                               'Cumulative Fuel Consumption (gal)'])
            #     # df_outage.to_csv(os.path.join(results_dir, "February_{}_{}_{}_discount_{}.csv".format(site_name, scenario_choice, atb_year, discount_rate)))
            # df_outage.to_csv(savepath + "DIESEL_Outputs.csv")







        
if __name__ == "__main__":
    forced_diesel_sizes=False
    force_diesel_cat=False #only used if forced_diesel_sizes is True and generator type is 'CAT' or 'Both'
    force_cat_size_kw=5000 #only used if generator type is 'CAT' or 'Both'
    force_diesel_warsilla=False #only used if generator type is 'WRS' or 'Both'
    force_wrs_size_kw=10000 
    diesal_input={'forced_diesel_sizes':forced_diesel_sizes,'force_diesel_cat':force_diesel_cat,'force_diesel_wrs':force_diesel_warsilla,'force_cat_size_kw':force_cat_size_kw,'force_wrs_size_kw':force_wrs_size_kw}
    df = pd.read_csv('/Users/egrant/Desktop/MRCL_Things/Total_Loads.csv', skiprows=0, usecols = ['Total (kWh)'])
    load=df['Total (kWh)']
    load=load[892:899].to_list()
    gen_type='CAT' #options: 'CAT' or 'WRS' or 'Both'
    diesal_input.update({'Shortfall':load,'Generator Type(s)':gen_type})
    diesal_output={}
    gen_model=SimpleDiesel(diesal_input,diesal_output)
    []
    final_sf, final_power_supplied, final_fuel_consump, total_cost=gen_model.run_diesel()
    cost,performance,specs=gen_model.reorg_output_dict()
    []
    gen_model.post_process_diesel([gen_type],savepath='')
   


