'''
Test that people are in valid states
'''

import numpy as np
import sciris as sc
import fpsim as fp

def test_states():
    ''' Testing states '''
    sc.heading('Testing model states...')

    # Set up
    pars = fp.pars('test')
    exp = fp.ExperimentVerbose(pars)
    exp.run_model(mother_ids=True)
    res = exp.total_results

    # Checks that:
    #     no one is SA and pregnant
    #     lam coincides with breastfeeding
    #     everyone who is breastfeeding is lactating
    #     everyone who is gestating is pregnant (gestation > 0)
    print('Testing state overlaps...')
    preg_count = 0
    gestation = {}
    feed_lact = 0
    lam_lact = 0
    lact_lam = 0
    lact_total = 0
    lam_total = 0
    gest_not_preg = 0

    postpartum = {}

    postpartum_dur = {}

    for year, attribute_dict in res.items():
        pregnant = attribute_dict['pregnant']
        breastfeeding = attribute_dict['breastfeed_dur']
        lactating = attribute_dict['lactating']
        lam = attribute_dict['lam']
        gestation_list = attribute_dict['gestation']
        postpartum[year] = attribute_dict['postpartum']
        postpartum_dur[year] = [attribute_dict['postpartum_dur'][index] for index in [i for i, x in enumerate(attribute_dict['postpartum']) if x]]

        for index, preg_value in enumerate(pregnant):
            if breastfeeding[index] > 0:
                if not lactating[index]:
                    feed_lact = feed_lact + 1
            if lam[index]:
                lam_total = lam_total + 1
                if not lactating[index]:
                    lam_lact = lam_lact + 1
            if lactating[index]:
                lact_total = lact_total + 1
                if lam[index]:
                    lact_lam = lact_lam + 1
            if preg_value:
                preg_count = preg_count + 1
                gestation[index] = gestation_list[index]
            if gestation_list[index] > 0:
                if not preg_value:
                    gest_not_preg = gest_not_preg + 1

    descriptions = {0: "were breastfeeding while not lactating", 1: "were gestating while not pregnant"}
    for index, count_check in enumerate([feed_lact, gest_not_preg]):
        assert count_check == 0, f"{count_check} {descriptions[index]}"


    # Checks that no dead people are updating any parameters, specifically gestation and breastfeeding
    print('Testing dead people...')
    alive_recorder = {}
    gestation_dur = {}
    breastfeed_dur = {}

    for year, attribute_dict in res.items():
        for index, value in enumerate(attribute_dict["alive"]):
            if index not in alive_recorder:
                alive_recorder[index] = []
                gestation_dur[index] = []
                breastfeed_dur[index] = []
            alive_recorder[index].append(value)
            gestation_dur[index].append(attribute_dict["gestation"][index])
            breastfeed_dur[index].append(attribute_dict["breastfeed_dur"][index])

    prec_gestation = 100
    prec_breastfeed = 100
    for person in gestation_dur:
        for index, compared_value in enumerate(gestation_dur[person]):
            assert not(prec_gestation < compared_value and not alive_recorder[person][index-1]), "At [{i}, {index}] a person's pregnancy is progressing while they are dead"
            assert not(prec_breastfeed < breastfeed_dur[person][index] and not alive_recorder[person][index-1]), "At [{i}, {index}] a person is breastfeeding while they are dead"

    # Checks that lactation, gestation, postpartum do not preclude pregnancy
    print('Testing pre-pregnancy...')
    was_pregnant = {}
    for year, attribute_dict in res.items():
        for index, value in enumerate(attribute_dict["pregnant"]):
            if index not in was_pregnant or value:
                was_pregnant[index] = value

            gestation = attribute_dict["gestation"][index]
            lactating = attribute_dict["lactating"][index]
            postpartum = attribute_dict["postpartum"][index]
            assert not((gestation > 0 or lactating or postpartum > 0) and not was_pregnant[index]), f"In year {year} there was a person whose gestation is {gestation} lactation is {lactating} and postpartum is {postpartum} and their was_pregnant status is {was_pregnant[index]}"

    # Checks that age at first birth is consistent with dobs and ages
    print('Testing first birth age...')
    last_year_lengths = [len(dob) for dob in res[min(res.keys())]['dobs']] # get first value of dobs to initialize
    for year, attribute_dict in res.items():
        age_first_birth = attribute_dict['first_birth_age']
        ages = attribute_dict['age']
        this_year_lengths = [len(dob) for dob in attribute_dict['dobs']]
        # dobs and age at first birth should be consistent (specifically checks that mothers represented in first_birth_age is subset of those represented in dobs)
        for index, last_year_length in enumerate(last_year_lengths):
            if last_year_length == 0 and this_year_lengths[index] == 1:
                assert np.isclose(ages[index], age_first_birth[index], atol=0.1), f"Age at first birth is {ages[index]} but recorded as {age_first_birth[index]}"
        last_year_lengths = sc.dcp(this_year_lengths)

    # Checks that sexual debut and sexual debut age are consistent with sexually active and ages respectively
    print('Testing sexual debut...')
    all_sa = set([index for index, value in enumerate(res[min(res.keys())]['sexually_active']) if value])
    for year, attribute_dict in res.items():
        ages = attribute_dict['age']
        sexual_debut = attribute_dict['sexual_debut']
        sexual_debut_age = attribute_dict['sexual_debut_age']
        sexually_active_indices = set([index for index, value in enumerate(attribute_dict['sexually_active']) if value])
        newly_sexually_active = sexually_active_indices - all_sa
        for index in newly_sexually_active:
            assert sexual_debut[index], "Person is newly sexually active but not marked as such in sexual debut list"
            assert np.isclose(sexual_debut_age[index], ages[index], atol=0.1), "Age of person at sexual_debut_age doesn't match age when newly sexually active"
        all_sa = all_sa | sexually_active_indices

    # Checks that people under 11 or over 45 can't get pregnant
    print('Testing age boundaries...')
    for year, attribute_dict in res.items():
        for index, pregnant_bool in enumerate(attribute_dict["pregnant"]):
            age = attribute_dict['age'][index]
            if age < 10 or age > 51:
                assert not pregnant_bool, f"Individual {index} can't be pregnant she's {age}"


    # Checks that ages aren't wrong due to rounding the months incorrectly
    print('Testing ages...')
    for year, attribute_dict in res.items():
        if year in sorted(res.keys())[-10:]:
            for individual, age in enumerate(attribute_dict['age']):
                age_year = int(age)
                month = (age - age_year)
                assert np.isclose(month * 12, round(month * 12), atol=0.5), f"Individual at index: {individual} in year {year} has an age of {age} with a month ({month}) that is not a multiple of 1/12. month * 12 = {month * 12}"

    return exp


# Run tests
if __name__ == '__main__':
    with sc.timer():
        exp = test_states()