import sys
# Ugly!
sys.path.insert(0, "/home/dklein/GIT/DtkTrunk/Scripts/serialization")

import pprint
import dtkFileTools as dft

# state file from https://comps.idmod.org/#explore/Simulations?filters=Id=3aa0faf8-7306-ea11-a2c3-c4346bcb1551&offset=0&count=100&selectedId=3aa0faf8-7306-ea11-a2c3-c4346bcb1551
data = dft.read('state-00059.dtk')

print(f'This serialization contains {len(data.nodes)} node(s)')
for idx, node in enumerate(data.nodes):
    print(f'* Node {idx} contains {len(node.individualHumans)} particle(s)')
print('HEADER:')
for k,v in data.header.items():
    print(f'* {k} : {v}')

pp = pprint.PrettyPrinter(indent=2)
#pp.pprint(data.nodes[0].individualHumans[0])

# Find an individual on PILL

for i in range(100):
    print('-'*80)
    h = next(h for h in data.nodes[0].individualHumans if 'PILL' in h['Properties'][0])
    #h = next(h for h in data.nodes[0].individualHumans if 'IMPLANT' in h['Properties'][0])
    pp.pprint(h)

exit()

for p in data.nodes[0].individualHumans:
    print(f"Individual with id = {p['suid']['id']}")
    if 'interventions' in p:
        ic = p['interventions']
        print('birth_rate_mod:', ic['birth_rate_mod'])

        for intv in ic['interventions']:
            if intv['__class__'] == 'Contraceptive':
                print('currentEffect', intv['m_pWaningEffect']['currentEffect'])


