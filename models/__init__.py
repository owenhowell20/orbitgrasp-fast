import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

equiformer_path = '../models/equiformer_v2'
sys.path.append(equiformer_path)
