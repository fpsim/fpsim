"""
Defines a family planning module for use in Starsim
"""

import starsim as ss
import numpy as np
import sciris as sc


# %% FP module
 
class FPmod(ss.Demographics):

    def __init__(self, pars=None, **kwargs):
        super().__init__(name='fpmod')

        # Create parameters
        self.default_pars(
            fecundity_max=50,

            # Probabilities: can all be overwritten by age/time-varying values
            p_conceive=0.1,
            p_conceive_dist=ss.bernoulli(p=0),  # Placeholder: monthly probability of conceiving
            p_stillbirth=ss.bernoulli(p=0.02),  # Probability of a stillbirth
            p_abortion=ss.bernoulli(p=0.01),    # Probability of abortion
            p_twins=ss.bernoulli(p=0.001),
            p_maternal_mortality=ss.bernoulli(p=0.01),
            p_miscarriage=ss.bernoulli(p=0.1),
            p_contra=ss.bernoulli(p=0.2),
            miscarriage_timing=ss.choice(
                a=np.arange(9),
                p=np.array([0.7, 0.2, 0.07, 0.025, 0.001, 0.001, 0.001, 0.001, 0.001])
            ),

            # Durations
            dur_pregnancy=ss.uniform(7, 9),
            dur_postpartum=ss.constant(6),
            dur_lactation=ss.constant(6),
            dur_contra=ss.lognorm_ex(12, 24),   # Overall average duration of contraception use

            # Other
            eff_contra=0.7,       # Overall average efficacy of contraception
            sex_ratio=ss.bernoulli(0.5),
            rel_fecundity=ss.normal(1, 0.1)     # Individual variation in fecundity
        )
        self.update_pars(pars, **kwargs)

        # Create states
        self.add_states(
            # Contraception
            ss.BoolArr('on_contra'),        # Whether she's on contraception
            ss.FloatArr('method'),          # Which method to use - 0 used for those on no method
            ss.FloatArr('ti_contra'),       # Time point at which to set method
            ss.BoolArr('ever_contra'),      # Ever been on contraception

            # Counts of major birth/fertility-related events
            ss.FloatArr('parity'),          # Number of births including stillbirths
            ss.FloatArr('n_live_births'),   # Number of live births
            ss.FloatArr('n_stillbirths'),   # Number of stillbirths
            ss.FloatArr('n_miscarriages'),  # Number of miscarriages
            ss.FloatArr('n_abortions'),     # Number of abortions
            ss.FloatArr('n_pregnancies'),   # Number of pregnancies: should equal to the sum of the above events

            # Sexual and reproductive states
            ss.BoolArr('fertile', default=True),
            ss.BoolArr('pregnant'),         # Whether pregnant
            ss.BoolArr('postpartum'),       # Whether postpartum
            ss.BoolArr('sexually_active'),  # Whether currently sexually active
            ss.BoolArr('sexual_debut'),     # Whether past sexual debut
            ss.BoolArr('lactating'),        # Whether she's lactating
            ss.BoolArr('mother'),           # Whether she's a mother

            # Timesteps of significant events
            ss.FloatArr('ti_pregnant'),
            ss.FloatArr('ti_delivery'),
            ss.FloatArr('ti_postpartum'),
            ss.FloatArr('ti_miscarriage'),
            ss.FloatArr('ti_stop_postpartum'),
            ss.FloatArr('ti_stop_lactation'),
            ss.FloatArr('ti_sexually_inactive'),
            ss.FloatArr('ti_sexually_active'),
            ss.FloatArr('ti_dead'),

            # Durations
            ss.FloatArr('dur_pregnancy'),

            # States to track significant ages
            ss.FloatArr('sexual_debut_age'),    # Age of sexual debut -- but see comment below
            ss.FloatArr('fated_debut_age'),     # Age of 'fated' debut - might differ from above if she debuts but isn't immediately active (??)
            ss.FloatArr('first_birth_age'),     # Age at first birth

            # Fecundity
            ss.FloatArr('fecundity'),
        )

        # Any other initialization
        self.choose_slots = None  # Distribution for choosing slots; set in self.initialize()

        return

    def init_pre(self, sim):
        super().init_pre(sim)
        low = sim.pars.n_agents + 1
        high = int(sim.pars.slot_scale*sim.pars.n_agents)
        self.choose_slots = ss.randint(low=low, high=high, sim=sim, module=self)
        return

    def init_post(self):
        super().init_post()
        return

    def init_results(self):
        """ Initialize results """
        super().init_results()
        npts = self.sim.npts
        self.results += ss.Result(self.name, 'conceptions', npts, dtype=int, scale=True)
        self.results += ss.Result(self.name, 'pregnancies', npts, dtype=int, scale=True)
        self.results += ss.Result(self.name, 'births', npts, dtype=int, scale=True)
        self.results += ss.Result(self.name, 'deaths', npts, dtype=int, scale=True)
        self.results += ss.Result(self.name, 'live_births', npts, dtype=int, scale=True)
        self.results += ss.Result(self.name, 'stillbirths', npts, dtype=int, scale=True)
        self.results += ss.Result(self.name, 'miscarriages', npts, dtype=int, scale=True)
        self.results += ss.Result(self.name, 'abortions', npts, dtype=int, scale=True)
        self.results += ss.Result(self.name, 'maternal_deaths', npts, dtype=int, scale=True)
        self.results += ss.Result(self.name, 'infant_deaths', npts, dtype=int, scale=True)
        self.results += ss.Result(self.name, 'mcpr', npts, dtype=int, scale=False)
        self.results += ss.Result(self.name, 'cpr', npts, dtype=int, scale=False)

        return

    def update(self):
        """ Perform all updates """
        self.update_states()  # Delivery, miscarriages, postpartum states
        self.update_methods()  # Contraception
        preg_uids = self.make_pregnancies()  # Conception
        self.make_embryos(preg_uids)  # Set properties for newly-conceived agents
        return

    def update_states(self):
        """ Update states """
        ti = self.sim.ti
        ppl = self.sim.people
        pars = self.pars

        # Update fertility
        self.fertile[ppl.male] = False
        self.fertile[(ppl.age > pars.fecundity_max)] = False

        # Trigger delivery
        deliv_uids = (self.pregnant & (self.ti_delivery <= ti)).uids
        if len(deliv_uids):
            self.trigger_delivery(deliv_uids)
            self.results.births[ti] += len(deliv_uids)

        # Trigger miscarriages
        mis_uids = (self.pregnant & (self.ti_miscarriage <= ti)).uids
        if len(mis_uids):
            self.trigger_miscarriage(mis_uids)
            self.results.miscarriages[ti] += len(mis_uids)

        # Update postpartum, breastfeeding, and LAM state
        pp_uids = (self.postpartum & (self.ti_stop_postpartum <= ti)).uids
        self.postpartum[pp_uids] = False
        self.fertile[pp_uids] = True

        return

    def update_methods(self):
        """ Update women's contraceptive choices and methods """
        ti = self.sim.ti

        # Stopping contraception
        stopping_uids = (self.on_contra & (self.ti_contra <= ti)).uids
        if len(stopping_uids):
            self.on_contra[stopping_uids] = False

        # Starting contraception
        off_contra_uids = (~self.on_contra & self.sim.people.female).uids
        starting_contra = self.pars.p_contra.filter(off_contra_uids)
        self.on_contra[starting_contra] = True
        dur_contra = self.pars.dur_contra.rvs(starting_contra)
        self.ti_contra[starting_contra] = ti + dur_contra

        return

    def stop_pregnancy(self, uids):
        """ Helper function to reset states once a pregnancy stops for any reason """
        self.pregnant[uids] = False
        self.ti_contra[uids] = self.sim.ti + 1  # Trigger a call to re-evaluate whether to use contraception
        return

    def trigger_delivery(self, deliv_uids):
        """ Trigger delivery for women who've reached the end of their pregnancies """
        ti = self.sim.ti

        # Update states
        self.stop_pregnancy(deliv_uids)
        self.postpartum[deliv_uids] = True

        # Determine times of postpartum events
        dur_postpartum = self.pars.dur_postpartum.rvs(deliv_uids)
        self.ti_stop_postpartum[deliv_uids] = ti + dur_postpartum

        # Handle stillbirth
        still_uids, live_uids = self.pars.p_stillbirth.split(deliv_uids)
        if len(still_uids):
            self.n_stillbirths[still_uids] += 1  # Track how many stillbirths an agent has had
            self.results.stillbirths[ti] = len(still_uids)
            self.parity[still_uids] += 1

        # Determine outcomes for live births: twins, mortality, lactation
        if len(live_uids):
            # Twins
            twin_uids, single_uids = self.pars.p_twins.split(live_uids)
            self.n_live_births[single_uids] += 1
            self.n_live_births[twin_uids] += 2
            self.parity[single_uids] += 1
            self.parity[twin_uids] += 2
            self.results.live_births[ti] = len(single_uids) + 2*len(twin_uids)

            # Mortality
            maternal_deaths = self.pars.p_maternal_mortality.filter(live_uids)
            if len(maternal_deaths):
                self.sim.people.request_death(maternal_deaths)

            # Lactation
            self.lactating[live_uids] = True
            dur_lactation = self.pars.dur_lactation.rvs(live_uids)
            self.ti_stop_lactation[live_uids] = ti + dur_lactation

        return

    def trigger_miscarriage(self, mc_uids):
        """ Trigger miscarriage """
        self.stop_pregnancy(mc_uids)
        self.n_miscarriages[mc_uids] += 1
        return

    def make_pregnancies(self):
        """ Check for new conceptions """
        ti = self.sim.ti

        # Greatly simplified version, need to add more logic here
        p_conceive = np.zeros(len(self.sim.people))
        p_conceive[self.fertile & self.on_contra] = self.pars.p_conceive
        p_conceive[self.fertile & ~self.on_contra] = self.pars.p_conceive * self.pars.eff_contra
        self.pars.p_conceive_dist.set(p=p_conceive[self.fertile])
        conceives_uids = self.pars.p_conceive_dist.filter(self.fertile.uids)
        self.results.conceptions[ti] += len(conceives_uids)
        self.n_pregnancies[conceives_uids] += 1

        # Decide whether to continue the pregnancy
        abort_uids, preg_uids = self.pars.p_abortion.split(conceives_uids)
        if len(abort_uids):
            self.n_abortions[abort_uids] += 1
            self.results.abortions[ti] += len(abort_uids)
        if len(preg_uids):
            self.set_prognoses(preg_uids)

        return preg_uids

    def set_prognoses(self, uids):
        """ For people who've just conceived, determine the outcomes and times """
        ti = self.sim.ti

        # Update states
        self.pregnant[uids] = True
        self.postpartum[uids] = False
        self.lactating[uids] = False

        # Handle miscarriage
        mis_uids, ft_uids = self.pars.p_miscarriage.split(uids)
        if len(mis_uids):
            dur_pregnancy = self.pars.miscarriage_timing.rvs(mis_uids)
            self.ti_miscarriage[mis_uids] = ti + dur_pregnancy

        # Set delivery date for those who don't miscarry
        if len(ft_uids):
            dur_pregnancy = self.pars.dur_pregnancy.rvs(ft_uids)
            self.dur_pregnancy[ft_uids] = dur_pregnancy
            self.ti_delivery[ft_uids] = ti + dur_pregnancy

        return

    def make_embryos(self, conceive_uids):
        """ Add properties for the just-conceived """
        people = self.sim.people
        n_unborn = len(conceive_uids)
        if n_unborn == 0:
            new_uids = ss.uids()
        else:

            # Choose slots for the unborn agents
            new_slots = self.choose_slots.rvs(conceive_uids)

            # Grow the arrays and set properties for the unborn agents
            new_uids = people.grow(len(new_slots), new_slots)
            people.age[new_uids] = -self.dur_pregnancy[conceive_uids]
            people.slot[new_uids] = new_slots
            people.female[new_uids] = self.pars.sex_ratio.rvs(conceive_uids)

            # Add connections to maternal transmission layers. This is optional,
            # and it should be possible to run the model without networks.
            for lkey, layer in self.sim.networks.items():
                if layer.prenatal:
                    durs = self.dur_pregnancy[conceive_uids]
                    start = np.full(n_unborn, fill_value=self.sim.ti)
                    layer.add_pairs(conceive_uids, new_uids, dur=durs, start=start)

        return new_uids


# %% Test run module

if __name__ == '__main__':

    sc.heading('Test FPmod Starsim run')
    fpmod = FPmod()
    par_kwargs = dict(n_agents=500, start=2000, end=2010, rand_seed=1, verbose=1)
    sim = ss.Sim(demographics=[fpmod, ss.Deaths()], networks='maternal', **par_kwargs)
    sim.run()
