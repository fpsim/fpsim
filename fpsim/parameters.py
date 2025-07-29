'''
Handle sim parameters
'''

import sciris as sc
import starsim as ss
from . import defaults as fpd

__all__ = ['SimPars', 'FPPars', 'make_sim_pars', 'make_fp_pars', 'par_keys', 'sim_par_keys']


# %% Parameter creation functions


class SimPars(ss.SimPars):
    """
    Dictionary with all parameters used within an FPsim.
    All parameters that don't vary across geographies are defined explicitly here.
    Keys for all location-specific parameters are also defined here with None values.
    """
    def __init__(self, **kwargs):

        # Initialize the parent class
        super().__init__()
        self.n_agents = 1_000  # Number of agents
        self.pop_scale = None   # Scaled population / total population size
        self.start = 1960   # Start year of simulation
        self.stop = 2020   # End year of simulation
        self.dt = 1/12      # The simulation timestep in 'unit's
        self.unit = 'year'   # The unit of time for the simulation
        self.rand_seed = 1      # Random seed
        self.verbose = 1/12   # Verbosity level
        self.use_aging = True   # Whether to age the population
        # Update with any supplied parameter values and generate things that need to be generated
        self.update(kwargs)
        return


def make_sim_pars(**kwargs):
    """ Shortcut for making a new instance of SimPars """
    return SimPars(**kwargs)


class FPPars(ss.Pars):
    def __init__(self, **kwargs):
        super().__init__()

        # Basic parameters
        self.location = None   # CONTEXT-SPECIFIC ####

        # Settings - what aspects are being modeled - TODO, remove
        self.use_partnership = 0

        # Age limits (in years)
        self.method_age = 15
        self.age_limit_fecundity = 50
        self.max_age = 99

        # Durations (in months)
        self.end_first_tri = 3      # Months
        self.preg_dur_low = 9       # Months
        self.preg_dur_high = 9      # Months
        self.max_lam_dur = 5        # Duration of lactational amenorrhea (months)
        self.short_int = 24         # Duration of a short birth interval between live births (months)
        self.low_age_short_int = 0  # age limit for tracking the age-specific short birth interval
        self.high_age_short_int = 20    # age limit for tracking the age-specific short birth interval
        self.postpartum_dur = 35    # Months
        self.breastfeeding_dur_mean = None  # CONTEXT-SPECIFIC #### - Parameter of truncated norm distribution
        self.breastfeeding_dur_sd = None    # CONTEXT-SPECIFIC #### - Parameter of truncated norm distribution

        # Pregnancy outcomes
        self.abortion_prob = None   # CONTEXT-SPECIFIC ####
        self.twins_prob = None   # CONTEXT-SPECIFIC ####
        self.LAM_efficacy = 0.98   # From Cochrane review: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6823189/
        self.maternal_mortality_factor = 1

        # Fecundity and exposure
        self.fecundity_var_low = 0.7
        self.fecundity_var_high = 1.1
        self.primary_infertility = 0.05
        self.exposure_factor = 1.0    # Overall exposure correction factor

        # Other sim parameters
        self.mortality_probs = {}

        ###################################
        # Context-specific data-derived parameters, all defined within location files
        ###################################
        self.filenames = None
        self.age_pyramid = None
        self.age_mortality = None
        self.maternal_mortality = None
        self.infant_mortality = None
        self.miscarriage_rates = None
        self.stillbirth_rate = None
        self.age_fecundity = None
        self.fecundity_ratio_nullip = None
        self.lactational_amenorrhea = None
        self.sexual_activity = None
        self.sexual_activity_pp = None
        self.debut_age = None
        self.exposure_age = None
        self.exposure_parity = None
        self.spacing_pref = None
        self.barriers = None
        self.urban_prop = None
        self.wealth_quintile = None
        self.age_partnership = None
        self.mcpr = None
        self.region = None
        self.track_children = False  # Whether to track children
        self.regional = None

        self.update(kwargs)
        self.update_location()

        return

    def update_location(self):
        """
        Update the location-specific parameters based on the current location.
        """
        if self.location is None:
            self.location = fpd.get_location(self.location, printmsg=True)  # Handle location

        # Import the location module
        from . import locations as fplocs

        # Use external registry for locations first
        if self.location in fpd.location_registry:
            location_module = fpd.location_registry[self.location]
            location_pars = location_module.make_pars()
        elif hasattr(fplocs, self.location):
            location_pars = getattr(fplocs, self.location).make_pars()
        else:
            raise NotImplementedError(f'Could not find location function for "{self.location}"')

        self.update(**location_pars)
        return


def make_fp_pars():
    """ Shortcut for making a new instance of FPPars """
    return FPPars()


# Shortcut for accessing default keys
par_keys = make_fp_pars().keys()
sim_par_keys = make_sim_pars().keys()
