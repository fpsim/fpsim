"""
Calibrate Kenya using the simple version of method choice
"""
import fpsim as fp
import sciris as sc

#
def make_run_sim(calib_pars=None):
    method_choice = fp.SimpleChoice(location='kenya')
    sim = fp.Sim(pars=calib_pars, contraception_module=method_choice)
    sim.run()
    return sim


def plot_results():
    return

if __name__ == '__main__':

    do_run = True
    do_save = True
    do_plot = True
    to_plot = []

    calib_pars = dict(
        fecundity_var_low=1,
        fecundity_var_high=1.27,
        exposure_factor=1.93,
    )

    if do_run:
        sim = make_run_sim()
        if do_save:
            sc.saveobj('results/kenya.sim', sim)

    if do_plot:
        plot_results()

