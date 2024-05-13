import numpy as np

from .utils import create_state_space

parameters = {
    'moving_arm': 'right',
    
    'x_reaching_space_limits': (-300, 300),
    'y_reaching_space_limits': (0, 300),

    'x_step_size': 20.,
    'y_step_size': 20.,

    'starting_points': [(-200, 50), (-100, 50), (-200, 150), (100, 50), (200, 50)],
    'reaching_points': [(-250, 150), (-200, 150), (150, 250), (-100, 150), (-50, 150),
                        (0, 150),  (50, 150), (100, 150),  (150, 150), (200, 150), (250, 150)],

    'dim_motor': 22
}

parameters['motor_orientations'] = np.linspace(0, 360, parameters['dim_motor'])

parameters['sig_s1'] = 50.  # in [mm]
parameters['sig_pm'] = 200.  # in [mm]

parameters['dim_bg'] = parameters['dim_motor']

parameters['strength_efference_copy'] = 0.75
parameters['sig_m1'] = 15  # in [°]

state_space = create_state_space(
    x_bound=parameters['x_reaching_space_limits'],
    y_bound=parameters['y_reaching_space_limits'],
    step_size_x=parameters['x_step_size'],
    step_size_y=parameters['y_step_size'],
)

parameters['dim_s1'] = state_space.shape[:-1]
parameters['dim_str'] = tuple(list(parameters['dim_s1']) + [25])

