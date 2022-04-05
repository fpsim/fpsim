import fpsim as fp
from collections import defaultdict
from . import defaults as fpd
import sciris as sc
import pandas as pd
from copy import deepcopy

class SimVerbose(fp.Sim):
    def __init__(self, pars=None, mother_ids=False):
        """
        Initializes a SimVerbose object which extends the logging functionality of a normal fp.Sim for the sake of testing

        Configurable:
            self.to_csv::bool:
                Saves a csv file for each state selected in the format of row: individual, column: timestep, value: state
            self.custom_csv_tables::bool:
                If the user is saving a csv, this is a list of the states to record csv files for (default is all states)
            self.to_file::bool:
                Save the results of the sim (either json or multiple csv's)
        """
        super().__init__(pars, mother_ids=mother_ids)
        self.test_mode = True
        self.to_csv = False
        self.custom_csv_tables = None
        self.to_file = False
        self.initialized = False

        self.total_results = defaultdict(lambda: {})

        self.last_year_births = [0] * pars['n']
        self.last_year_gestations = [0] * pars['n']
        self.last_year_sexactive = [0] * pars['n']
        self.last_year_deaths = [0] * pars['n']
        self.dead_moms = set()
        self.is_sexactive = set()
        self.events = defaultdict(dict)
        self.channels = ["Births", "Conceptions", "Miscarriages", "Sexual_Debut", "Deaths"]

    def log_daily_totals(self):
        """
        Logs data for total_results and events at each timestep.

        Output:
            self.total_results::dict
                Dictionary of all individual results formatted as {timestep: attribute: [values]}
                keys correspond to fpsim.defaults debug_states
            self.events::dict
                Dictionary of events correponding to self.channels formatted as {timestep: channel: [indices]}.
        """
        for state in fpd.debug_states:
            self.total_results[self.y][state] = getattr(self.people, state)

        # Getting births gestation and sexual_activity
        self.this_year_births = deepcopy(self.total_results[self.y]["parity"])
        self.this_year_gestations = deepcopy(self.total_results[self.y]["gestation"])
        self.this_year_sexactive = deepcopy(self.total_results[self.y]["sexually_active"])
        self.this_year_alive = deepcopy(self.total_results[self.y]["alive"])

        for channel in self.channels:
            self.events[self.y][channel] = []
        
        # Comparing parity of previous year to this year, adding births
        for index, last_parity in enumerate(self.last_year_births):
            if last_parity < self.this_year_births[index]:
                for i in range(self.this_year_births[index] - last_parity):
                    self.events[self.y]['Births'].append(index)

        # Comparing gestation of previous year to get conceptions and miscarriages
        for index, last_gestation in enumerate(self.last_year_gestations):
            if last_gestation < self.this_year_gestations[index] and last_gestation == 0:
                self.events[self.y]['Conceptions'].append(index)
            if last_gestation == (self.pars['end_first_tri'] - 1) and self.this_year_gestations[index] == 0:
                self.events[self.y]['Miscarriages'].append(index)
        
        # Getting first instance of a person being sexually active
        for index, active in enumerate(self.last_year_sexactive):
            if (not active) and (self.this_year_sexactive[index]) and (index not in self.is_sexactive):
                self.events[self.y]['Sexual_Debut'].append(index)
                self.is_sexactive.add(index)

        for index, alive in enumerate(self.last_year_deaths):
            if alive > self.this_year_alive[index]:
                self.events[self.y]['Deaths'].append(index)

        # Aggregate channels taken from people.results
        self.events[self.y]['Step_Results'] = self.people.step_results
        
        self.last_year_births = deepcopy(self.this_year_births)
        self.last_year_gestations = deepcopy(self.this_year_gestations)
        self.last_year_sexactive = deepcopy(self.this_year_sexactive)

    def save_daily_totals(self):
        """
        At the end of sim run, stores total_results as either a json or feather file.
    
        Inputs
            self.to_file::bool
                If True, writes results to file
            self.to_feather::bool
                If True, writes results to feather file
                If False, writes results to json file
        Outputs:
            Either a json file at "sim_output/total_results.json"
            or a feather file for each state at "sim_output/{state}_state"
        """
        if self.to_file:
            if not self.to_feather:
                sc.savejson(filename="sim_output/total_results.json", obj=self.total_results)
            else:
                if self.custom_csv_tables is None:
                    states = fpd.debug_states
                else:
                    states = self.custom_csv_tables
                for state in states:
                    state_frame = pd.DataFrame()
                    max_length = len(self.total_results[max(self.total_results.keys())][state])
                    for timestep, _ in self.total_results.items():
                        colname = str(timestep) + "_" + state
                        adjustment = max_length - len(self.total_results[timestep][state])
                        state_frame[colname] = list(self.total_results[timestep][state]) + [None] * adjustment # ONLY WORKS IF LAST YEAR HAS MOST PEOPLE

                    state_frame.to_csv(f"sim_output/{state}_state.csv")

    def story(self, index):
        """
        Prints a story of all major events in an individual's life based on calculated SimVerbose channels,
        base Sim channels, and statistics calculated within the function such as year of birth of individual.

        Inputs:
            index::int:
                index of the individual, must be less than population
        Outputs:
            printed display of each major event in the individual's life
        """ 
        print(self.events.keys())

        2019.08333
        year = 19
        
        def format_timestep(timestep):
            year = int(timestep)
            print(timestep)
            month_index = round(((timestep) - year) * 12)
            month = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][month_index]
            return f"{month}, {year}"
        
        if len(self.events) == 0:
            print("Story function can only be used after sim is run. Try Experiment.run_model() first")
            return
        ages = self.people['age'] # Progresses even if dead
        last_year = max(self.total_results.keys())
        year_born = last_year - ages[index]
        print(last_year)
        print(year_born)
        print(ages[index])
        print(f"This is the story of individual {index} who was born in {format_timestep(year_born)}")

        event_response_dict = {"Births": "gives birth", "Conceptions": "conceives", "Miscarriages": "has a miscarriage", "Sexual_Debut": "has her sexual debut", "Deaths": "dies"}
        # result_response_dict = {"breastfeed_dur": "begins breastfeeding", "begins lactating": "is lactating", "lam": "is on LAM", "postpartum":
        #                          "is postpartum", "pregnant": "is pregnant", "sexually_active": "is sexually active", "parity": "", "method"}
        # Want to display events, start of breastfeeding, end of breastfeeding, start of lactating, end of lactating, start of LAM, end of LAM, start of pregnancy,
        # When pregnant display the cardinality of the birth, end of sexual activity.
        last_method = fpd.method_list[self.total_results[min(self.total_results.keys())]['method'][index]]
        for timestep in self.events:
            if timestep >= year_born:
                for new_channel in event_response_dict:
                    if index in self.events[timestep][new_channel]:
                        if new_channel == "Births":
                            print(f"{format_timestep(timestep)} individual {index} gives birth to child number {self.total_results[timestep]['parity'][index]}")
                        else:
                            print(f"{format_timestep(timestep)} individual {index} {event_response_dict[new_channel]}")
            new_method = fpd.method_list[self.total_results[timestep]['method'][index]]
            if new_method != last_method:
                print(f"{format_timestep(timestep)} individual {index} switched from {last_method} to {new_method}")
            last_method = new_method
