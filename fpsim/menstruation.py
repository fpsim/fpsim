"""
Heavy Menstrual Bleeding
Adds an agent state to proxy heavy menstrual bleeding and initializes state 
"""
import fpsim as fp
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss


class Menstruation(ss.Connector):
    """
    Class to handle menstruation related events
    """
    
    def __init__(self, pars=None, name='menstruation', **kwargs):
        super().__init__(name=name)

        # Define parameters
        self.define_pars(
            unit='month',
            
            # Menses
            age_menses=ss.lognorm_ex(14, 3),  # Age of menarche
            age_menopause=ss.normal(50, 3),  # Age of menopause
            eff_hyst_menopause=ss.normal(-5, 1),  # Adjustment for age of menopause if hysterectomy occurs

            # The probability of IUD usage is set within FPsim, so this parameter just
            # determines whether each woman is a hormonal or non-hormonal IUD user
            p_hiud=ss.bernoulli(p=0.5),

            # HMB prediction
            p_hmb_prone=ss.bernoulli(p=0.4),  # Proportion of menstruating women who experience HMB (sans interventions)
            hmb_pred=sc.objdict(  # Parameters for HMB prediction
                base=0.95,  # For those prone to HMB, probability they'll experience it this timestep
                pill=-3,  # Effect of hormonal pill on HMB - placeholder
                hiud=-10,  # Effect of IUD on HMB - placeholder
            ),

            # Non-permanent sequelae of HMB
            hmb_seq=sc.objdict(
                poor_mh=sc.objdict(  # Parameters for poor menstrual hygiene
                    base=0.4,  # Intercept for poor menstrual hygiene
                    hiud=-0.5,  # Effect of IUD on poor menstrual hygiene - placeholder ##TODO: Allow for an effect of other hormonal contraception
                ),
                anemic=sc.objdict(  # Parameters for anemia
                    base=0.01,  # Baseline probability of anemia
                    hmb=1.5,  # Effect of HMB on anemia - placeholder
                ),
                pain=sc.objdict(  # Parameters for menstrual pain
                    base=0.1,  # Baseline probability of menstrual pain
                    hmb=1.5,  # Effect of HMB on menstrual pain - placeholder
                    hiud=-0.5,  # Effect of IUD on menstrual pain - placeholder ##TODO: Other contraceptive methods
                ),
            ),

            # Permanent sequelae of HMB
            hyst=sc.objdict(  # Parameters for hysterectomy
                base=0.01,  # Baseline probability of hysterectomy
                hmb=2,      # Effect of HMB on hysterectomy - placeholder
                lt40=-5,    # Adjustment for women less than 40
            ),
        )
        
        self.update_pars(pars, **kwargs)

        # Probabilities of various outcomes - all set via models within the module
        # Don't directly alter anything here, as the probabilities are calculated
        # via logistic regression models based on the parameters defined in the main
        # parameter dict.
        self._p_hmb = ss.bernoulli(p=0)
        self._p_poor_mh = ss.bernoulli(p=0)
        self._p_anemic = ss.bernoulli(p=0)
        self._p_pain = ss.bernoulli(p=0)
        self._p_hyst = ss.bernoulli(p=0)

        # Define states
        self.define_states(
            # HMB states
            ss.State('hmb_prone'),
            ss.State('hmb'),
            ss.State('hmb_sus', label="Susceptible to HMB"),

            # HMB sequelae
            ss.State('anemic'),
            ss.State('poor_mh', label="Poor menstrual hygiene"),
            ss.State('pain', label="Menstrual pain"),
            ss.State('hyst', label="Hysterectomy"),

            # Menstrual states
            ss.State('menstruating'),
            ss.State('premenarchal'),
            ss.State('post_menarche'),
            ss.State('menopausal'),
            ss.State('early_meno'),
            ss.State('premature_meno'),
            ss.FloatArr('age_menses', label="Age of menarche"),
            ss.FloatArr('age_menopause', label="Age of menopause"),

            # Contraceptive methods
            ss.State('pill', label="Using hormonal pill"),
            ss.State('hiud', label="Using hormonal IUD"),
            ss.State('hiud_prone', label="Prone to use hormonal IUD, if using IUD"),
        )

        return

    def init_results(self):
        """ Initialize results """
        super().init_results()
        results = [
            ss.Result('hmb_prev', scale=False, label="Prevalence of HMB"),
            ss.Result('poor_mh_prev', scale=False, label="Prevalence of poor menstrual hygiene"),
            ss.Result('anemic_prev', scale=False, label="Prevalence of anemia"),
            ss.Result('pain_prev', scale=False, label="Prevalence of menstrual pain"),
            ss.Result('hyst_prev', scale=False, label="Prevalence of hysterectomy"),
            ss.Result('hiud_prev', scale=False, label="Prevalence of IUD usage"),
            ss.Result('early_meno_prev', scale=False, label="Early menopause prevalence"),
            ss.Result('premature_meno_prev', scale=False, label="Premature menopause prevalence"),
        ]
        self.define_results(*results)
        return

    @property
    def lt40(self):
        return (self.sim.people.age < 40) & self.sim.people.female

    def _get_uids(self, upper_age=None):
        """ Get uids of females younger than upper_age """
        people = self.sim.people
        if upper_age is None: upper_age = 1000
        within_age = people.age < upper_age
        return (within_age & people.female).uids

    def set_mens_states(self, upper_age=None):
        """ Set menstrual states """
        f_uids = self._get_uids(upper_age=upper_age)
        self.age_menses[f_uids] = self.pars.age_menses.rvs(f_uids)
        self.age_menopause[f_uids] = self.pars.age_menopause.rvs(f_uids)
        self.hmb_prone[f_uids] = self.pars.p_hmb_prone.rvs(f_uids)
        self.hiud_prone[f_uids] = self.pars.p_hiud.rvs(f_uids)
        return

    def init_post(self):
        """ Initialize with sim properties """
        super().init_post()

        # Set initial menstrual states
        self.set_mens_states()
        self.set_hmb(self.hmb_sus.uids)

        return

    def _logistic(self, uids, pars):
        """ Calculate logistic regression probabilities """
        intercept = -np.log(1/pars.base-1)
        rhs = np.full_like(uids, fill_value=intercept, dtype=float)

        # Add all covariates
        for term, val in pars.items():
            if term != 'base':
                rhs += val * getattr(self, term)[uids]

        # Calculate the probability
        return 1 / (1+np.exp(-rhs))

    def set_hmb(self, uids):
        """ Set who will experience heavy menstrual bleeding (HMB) """
        # Calculate the probability of HMB
        p_hmb = self._logistic(uids, self.pars.hmb_pred)
        self._p_hmb.set(0)
        self._p_hmb.set(p_hmb)
        has_hmb = self._p_hmb.filter(uids)
        self.hmb[has_hmb] = True
        return

    def step(self):
        """ Updates for this timestep """

        # Set menstruating states
        self.set_mens_states(upper_age=self.t.dt)  # for new agents
        self.step_states()  # for existing agents

        mens_uids = self.menstruating.uids
        self.hmb[:] = False  # Reset

        # Update HMB
        self.set_hmb(self.hmb_sus.uids)

        # Set non-permanent sequalae of HMB
        for seq, p in self.pars.hmb_seq.items():
            old_attr = getattr(self, seq)
            old_attr[:] = False  # Reset the state
            setattr(self, seq, old_attr)  # Update the state
            attr_dist = getattr(self, f'_p_{seq}')
            attr_dist.set(0)

            # Calculate the probability of the sequelae
            p_val = self._logistic(mens_uids, p)
            attr_dist = getattr(self, f'_p_{seq}')
            attr_dist.set(p_val)
            has_attr = attr_dist.filter(mens_uids)
            new_attr = getattr(self, seq)
            new_attr[has_attr] = True
            setattr(self, seq, new_attr)

        # Set hysterectomy state
        hyst_sus = (self.menstruating & ~self.hyst).uids
        p_hyst = self._logistic(hyst_sus, self.pars.hyst)
        self._p_hyst.set(0)
        self._p_hyst.set(p_hyst)
        has_hyst = self._p_hyst.filter(hyst_sus)
        self.hyst[has_hyst] = True

        # For women who've had a hysterectomy, reset age of menopause
        eff_hyst_menopause = self.pars.eff_hyst_menopause.rvs(has_hyst)
        self.age_menopause[has_hyst] += eff_hyst_menopause

        return

    def step_states(self):
        """ Updates for this timestep """
        ppl = self.sim.people
        f = ppl.female
        self.premenarchal[:] = f & (ppl.age < self.age_menses)
        self.post_menarche[:] = f & (ppl.age > self.age_menses)
        self.menstruating[:] = f & (ppl.age <= self.age_menopause) & (ppl.age >= self.age_menses)
        self.menopausal[:] = f & (ppl.age > self.age_menopause)
        self.early_meno[:] = self.menopausal & (self.age_menopause < 45)
        self.premature_meno[:] = self.menopausal & (self.age_menopause < 40)
        self.hmb_sus[:] = self.menstruating & self.hmb_prone & ~self.hmb

        # Contraceptive methods
        pill_idx = ppl.contraception_module.get_method_by_label('Pill').idx
        iud_idx = ppl.contraception_module.get_method_by_label('IUDs').idx
        self.pill[:] = ppl.method == pill_idx
        self.hiud[:] = (ppl.method == iud_idx) & self.hiud_prone

        return

    def update_results(self):
        super().update_results()
        ti = self.ti
        def count(arr): return np.count_nonzero(arr)
        def cond_prob(a, b): return sc.safedivide(count(a & b), count(b))
        for res in ['hmb', 'poor_mh', 'anemic', 'pain', 'hiud', 'hyst', 'early_meno', 'premature_meno']:
            self.results[f'{res}_prev'][ti] = cond_prob(getattr(self, res), self.menstruating)
        for res in ['hyst', 'early_meno', 'premature_meno']:
            self.results[f'{res}_prev'][ti] = cond_prob(getattr(self, res), self.post_menarche)
        return


class contra_hmb(ss.Intervention):
    def __init__(self, pars=None, eligibility=None, **kwargs):
        super().__init__(name='contra_hmb', eligibility=eligibility)
        self.define_pars(
            year=2026,  # When to apply the intervention
            prob=ss.bernoulli(p=0.2),  # Proportion of HMB-prone non-users who will accept
        )
        self.update_pars(pars, **kwargs)
        if eligibility is None:
            self.eligibility = lambda sim: (
                    sim.people.menstruation.hmb_prone &
                    sim.people.menstruation.menstruating &
                    ~sim.people.on_contra &
                    ~sim.people.pregnant)
        self.define_states(
            ss.State('intervention_applied', label="Received IUD through intervention"),
        )
        return

    @property
    def iud_idx(self):
        """ Get the index of the IUD method """
        return self.sim.people.contraception_module.get_method_by_label('IUDs').idx

    def step(self):
        if sim.t.now() == self.pars.year:
            # Print message
            print('Changing IUDs!')

            # Get women who accept the intervention
            elig_uids = self.check_eligibility()
            accept_uids = self.pars.prob.filter(elig_uids)

            # Adjust their contraception
            sim.people.method[accept_uids] = self.iud_idx
            sim.people.on_contra[accept_uids] = True
            sim.people.ever_used_contra[accept_uids] = True
            self.intervention_applied[accept_uids] = True
        return

# ---------------- TEST ----------------

# if __name__ == '__main__':

#     mens = Menstruation()

#     from education import Education
#     objective_data = pd.read_csv("data/edu_objective.csv")
#     attainment_data = pd.read_csv("data/edu_initialization.csv")
#     edu = Education(objective_data=objective_data, attainment_data=attainment_data)

#     sim = fp.Sim(location='kenya', connectors=[mens, edu], interventions=contra_hmb, start=2020, stop=2030)
#     sim.run(verbose=1/12)

#     # Plot education
#     import pylab as pl
#     t = sim.results.education.timevec
#     fig, axes = pl.subplots(2, 3, figsize=(20, 12))
#     axes = axes.ravel()

#     res_to_plot = ['mean_attainment', 'mean_objective', 'prop_completed', 'prop_in_school', 'prop_dropped']
#     sc.options(fontsize=16)

#     for i, res in enumerate(res_to_plot):
#         ax = axes[i]
#         r0 = sim.results.education[res]
#         ax.plot(t, r0)
#         ax.set_title(res)

#     all_props = [sim.results.education.prop_in_school,
#                  sim.results.education.prop_completed,
#                  sim.results.education.prop_dropped]

#     ax = axes[-1]
#     ax.stackplot(t, all_props, labels=['In school', 'Completed', 'Dropped'], alpha=0.8)
#     ax.set_title('All AGYW')
#     ax.legend()

#     sc.figlayout()
#     pl.show()






    
    
    
    
    
    
    
    
    
    
        
        
        
        

