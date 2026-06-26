## For vercel deployment DONT CHANGE THIS

import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from main import app  
