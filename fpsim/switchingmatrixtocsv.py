import fpsim as fp
import sciris as sc
import pandas as pd
import numpy as np

class StoreMethods(fp.Analyzer):
    def __init__(self, age_groups):
        super().__init__()
        self.age_groups = age_groups
        self.data = {age_group: [] for age_group in age_groups}
    
    def apply(self, sim):
        for age_group in self.age_groups:
            methods = sc.dcp(sim['methods']['adjusted'])
            self.data[age_group].append(methods)

def generate_random_matrix(size):
    """Generate a random switching matrix."""
    matrix = np.random.rand(size, size)
    matrix /= matrix.sum(axis=1)[:, np.newaxis]  # Normalize rows to sum to 1
    return matrix.tolist()

def run_simulation():
    # Define simulation parameters
    n_agents = 10000
    start_year = 2012
    end_year = 2030
    
    # Define age groups
    age_groups = ['<20', '20-24', '25-29', '30-34', '35+']

    # Generate switching matrices for each age group
    switching_matrices = {age_group: generate_random_matrix(len(age_groups)) for age_group in age_groups}

    # Initialize simulation with the analyzer
    analyzer = StoreMethods(age_groups)
    sim = fp.Sim(location='nuhdss', n_agents=n_agents, start_year=start_year, end_year=end_year, analyzers=[analyzer])
    
    # Run the simulation
    sim.run()

    # Access and save the data for each age group
    for age_group in age_groups:
        if analyzer.data[age_group]:
            methods_data = analyzer.data[age_group][-1]  # Get the last step data for the age group
            
            # Convert the methods data to a pandas DataFrame
            df = pd.DataFrame(methods_data)
            
            # Replace invalid characters in age_group for filename
            safe_age_group = age_group.replace('<', 'under_').replace('+', 'plus').replace('-', '_')

            # Save the DataFrame to a CSV file
            df.to_csv(f'methods_data_{safe_age_group}.csv', index=False)
            print(f"Data saved to methods_data_{safe_age_group}.csv")

# Run the simulation function
run_simulation()
