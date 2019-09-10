#!/usr/bin/env python
import os
emod_path = os.getenv('EMOD_PATH', 'emod')
main_file = 'param_overrides.json'
os.system(f'{emod_path} --config {main_file} -P .')
