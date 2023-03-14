import sys
import os
join = os.path.join
sys.path.append('.')
sys.path.append(join('GuidanceRewards'))

from simglucose.simulation.user_interface import simulate
from sac_controller import MyController
from simglucose.controller.basal_bolus_ctrller import BBController
from encoder import AE

ctrller = MyController(0)
# ctrller = BBController()
simulate(controller=ctrller)

