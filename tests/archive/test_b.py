import fpsim as fp
import sciris as sc


def test_to_df():
    sc.heading('Testing other sim methods...')

    sim = fp.Sim(location='test').run()
    sim.brief()

    df = sim.to_df()
    births = df.births.sum()
    last = df.t.values[-1]
    assert last == sim['end_year'], 'Last years do not match'
    assert births > 0, 'Expected births'

    return