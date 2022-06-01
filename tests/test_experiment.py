'''
Test births, conceptions, etc.
'''

import numpy as np
import sciris as sc
import fpsim as fp
import pytest

# Parameters
max_pregnancy_loss = 0.5 # Maximum allowed fraction of pregnancies to allow to not end in birth (including stillbirths)
do_plot  = 1 # Whether to do plotting in interactive mode
sc.options(backend='agg') # Turn off interactive plots


def ok(string):
    ''' Print out a successful test nicely '''
    return sc.printgreen(f'✓ {string}\n')


def test_channels():
    ''' Test Experiment channels '''
    sc.heading('Testing Experiment channels...')
    pars = fp.pars('test')
    exp = fp.ExperimentVerbose(pars)
    exp.run_model()
    events = exp.events
    channels = ["Births", "Conceptions", "Miscarriages", "Sexual_Debut", "Deaths"]

    # Checks that none of the channels from self.channels contain no entries.
    for channel in channels:
        if channel != "Deaths":
            maxval = 0
            for timestep in events:
                if len(events[timestep][channel]) > maxval:
                    maxval = len(events[timestep][channel])

            assert maxval > 0, f"Detected empty channel: {channel}"
    ok('No empty channels')

    # Checks that births (formatted as timestep: [indices]) is consistent with
    # the births aggregate value from step results (formatted as timestep: total).
    births = 0
    births_step = 0
    for timestep in events:
        births_step += events[timestep]["Step_Results"]["births"]
        births += len(events[timestep]['Births'])
    assert births == births_step, "Mismatch between step results births and births channel"
    ok(f'No mismatch for births ({births} == {births_step})')

    # Checks that conceptions is approximately births, and that conceptions is greater
    # than the number of births.
    births = 0
    conceptions = 0
    for timestep in events:
        births = births + len(events[timestep]['Births'])
        conceptions = conceptions + len(events[timestep]['Conceptions'])

    # We wouldn't expect more than a quarter of conceptions to end in miscarriages
    assert np.isclose(births, conceptions, atol=max_pregnancy_loss*births), "Less than 75 percent of conceptions result in births"
    assert conceptions > births, "Number of conceptions not greater than recorded births"

    # Checks that miscarriages < difference between conceptions and births
    births = 0
    conceptions = 0
    miscarriages = 0
    for timestep in events:
        births += len(events[timestep]['Births'])
        conceptions += len(events[timestep]['Conceptions'])
        miscarriages += len(events[timestep]['Miscarriages'])
        assert conceptions - births > miscarriages, "The number of miscarriages is greater than the differences between conceptions and births"
    ok(f'No mismatch for conceptions - births > miscarriages ({conceptions} - {births} > {miscarriages})')


    @pytest.mark.skip("Need to verify this works over multiple runs")
    def test_sexual_debut(self):
        """
        Checks that a person is SA at their sexual debut,
        and doesn't show up twice in the log of sexual debuts
        """
        sexually_active = set()
        for timestep in self.events:
            if timestep != 1960.0:
                for item in self.events[timestep]['Sexual_Debut']:
                    self.assertTrue(item not in sexually_active, msg=f"Record of individual at timestep {timestep} sexual debuted but not sexually active")
                    self.assertTrue(self.total_results[timestep]['sexually_active'][item], msg=f"Sexual debut but not sexually active at timestep {timestep} index {item}")
                sexually_active.update(self.events[timestep]['Sexual_Debut'])


def test_other():
    ''' Test other Experiment methods '''
    sc.heading('Testing other Experiment methods...')
    exp = fp.ExperimentVerbose(location='test').run()

    exp.to_json()
    ok('Experiment.to_json() succeeded')

    exp.sim.story(1)
    ok('SimVerbose.story() succeeded')

    return exp


def test_plot():
    ''' Test Experiment plotting '''
    sc.heading('Testing Experiment plotting...')
    if do_plot:
        pars = fp.pars('test')
        exp = fp.Experiment(pars)
        exp.run()
        exp.plot()
        ok('Plotting succeeded')
    return exp



if __name__ == '__main__':

    sc.options(backend=None) # Turn on interactive plots

    with sc.timer():
        exp1 = test_channels()
        exp2 = test_other()
        exp3 = test_plot()

