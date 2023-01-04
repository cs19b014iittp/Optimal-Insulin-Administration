from simglucose.simulation.user_interface import simulate
from sac_controller import MyController
from simglucose.controller.basal_bolus_ctrller import BBController

ctrller = MyController(0)
simulate(controller=ctrller)

