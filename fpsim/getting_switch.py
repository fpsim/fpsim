import fpsim as fp
import sciris as sc
import pandas as pd

class StoreMethods(fp.Analyzer):
    def __init__(self):
        super().__init__()
        self.data = dict()
    
    def apply(self, sim):
        methods = sc.dcp(sim['methods']['adjusted'])
        self.data[sim.i] = methods

def run_simulation():
    # Define simulation parameters
    n_agents = 10000
    start_year = 2012
    end_year = 2030

    # Initialize simulation with the analyzer
    analyzer = StoreMethods()
    sim = fp.Sim(location='nuhdss', n_agents=n_agents, start_year=start_year, end_year=end_year, analyzers=[analyzer])
    
    # Run the simulation
    sim.run()

    # Access the stored data for the 100th step
    time_point = 100
    if time_point in sim.pars['analyzers'][0].data:
        methods_data = sim.pars['analyzers'][0].data[time_point]
        
        # Convert the methods data to a pandas DataFrame
        df = pd.DataFrame(methods_data)
        
        # Save the DataFrame to a CSV file
        df.to_csv('methods_data_100th_step.csv', index=False)
        print("Data saved to methods_data_100th_step.csv")
    else:
        print(f"No data available for time point {time_point}")

if __name__ == '__main__':
    run_simulation()