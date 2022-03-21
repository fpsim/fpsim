import fpsim as fp

class ExperimentVerbose(fp.Experiment):
    def run_model(self, pars=None):
        """
        Create the sim and run the model, saving
        total results and individual events in the process
        """

        if not self.initialized:
            self.initialize()

        if pars is None:
            pars = self.pars

        self.sim = fp.SimVerbose(pars=pars)
        self.sim.run()
        self.post_process_sim()
        self.total_results = self.sim.total_results
        self.events = self.sim.events

        return
    