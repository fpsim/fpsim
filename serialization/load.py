import sys
sys.path.insert(0, "/home/dklein/GIT/DtkTrunk/Scripts/serialization")

import json
import dtkFileTools as dft

dtk = dft.read('output/state-00479.dtk')

print(dir(dtk))

print('NODES:\n', dir(dtk.nodes))
print('NODES:\n', dtk.nodes)
print('NODES:\n', dir(dtk.nodes.__parent__))
print('SIMULATION:\n', dtk.simulation.keys()) # ['serializationMask', 'infectionSuidGenerator', 'm_RngFactory', 'rng', 'campaignFilename', 'custom_reports_filename', 'sim_type', 'demographic_tracking', 'enable_spatial_output', 'enable_property_output', 'enable_default_report', 'enable_event_report', 'enable_node_event_report', 'enable_coordinator_event_report', 'enable_surveillance_event_report', 'loadbalance_filename']

print('SIMULATION:\n', dtk.simulation['serializationMask']) # ['serializationMask', 'infectionSuidGenerator', 'm_RngFactory', 'rng', 'campaignFilename', 'custom_reports_filename', 'sim_type', 'demographic_tracking', 'enable_spatial_output', 'enable_property_output', 'enable_default_report', 'enable_event_report', 'enable_node_event_report', 'enable_coordinator_event_report', 'enable_surveillance_event_report', 'loadbalance_filename']
print('HEADER:\n', dtk.header)
print('DATE:\n', dtk.date)
print('OBJECTS:\n', dir(dtk.objects))
print('CONTENTS:\n', dir(dtk.contents))
print('VERSION:\n', dtk.version)

print(dtk.nodes[0])

