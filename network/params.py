import numpy as np

from .utils import create_state_space

parameter_1D = {
    'x_reaching_space_limits': (-300, 300),
    'y_reaching_space_limits': (0, 300),

    'x_step_size': 10.,
    'y_step_size': 10.,

    'starting_points': [(-200, 50), (-100, 50), (0, 50), (100, 50), (200, 50)],
    'reaching_points': [(-250, 200), (-200, 200), (-150, 200), (-100, 200), (-50, 200),
                        (0, 200),  (50, 200), (100, 200),  (150, 200), (200, 200), (250, 200)],

    'dim_motor': 11
}

parameter_1D['motor_orientations'] = np.linspace(0, 180, parameter_1D['dim_motor'])

parameter_1D['dim_s1'] = parameter_1D['starting_points']
parameter_1D['sig_s1'] = 50.  # in [mm]
parameter_1D['dim_pm'] = parameter_1D['reaching_points']
parameter_1D['sig_pm'] = 10.  # in [mm]

parameter_1D['dim_str'] = (parameter_1D['dim_s1'], 20)
parameter_1D['dim_bg'] = (parameter_1D['dim_s1'], parameter_1D['dim_motor'])
