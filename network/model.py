from .params import parameters, state_space

from .connections import *
from .definitions import *

ann.setup(num_threads=6)

# input populations
PM = ann.Population(geometry=state_space.shape[:2], neuron=BaselineNeuron, name='PM')
S1 = ann.Population(geometry=parameters['dim_s1'], neuron=BaselineNeuron, name='S1')

SNc = ann.Population(geometry=1, neuron=DopamineNeuron, name='SNc')

# transmission populations into putamen
CM = ann.Population(geometry=parameters['dim_motor'], neuron=LinearNeuron, name='CM')
CM.tau = 20.
CM.noise = 0.01

# CBGT Loop (putamen)
StrD1 = ann.Population(geometry=parameters['dim_str'], neuron=StriatumD1Neuron, name='StrD1')
StrD1.noise = 0.0

GPe = ann.Population(geometry=parameters['dim_motor'], neuron=LinearNeuron, name='GPe')
GPe.noise = 0.01
GPe.baseline = 0.2

SNr = ann.Population(geometry=parameters['dim_bg'], neuron=SNrNeuron, name='SNr', stop_condition='r<0.1',)
SNr.noise = 0.01

VL = ann.Population(geometry=parameters['dim_bg'], neuron=LinearNeuron, name='VL')
VL.noise = 0.01
VL.baseline = 0.6

M1 = ann.Population(geometry=parameters['dim_bg'], neuron=LinearNeuron, name='M1')
M1.tau = 20.
M1.noise = 0.0
M1.baseline = 0.0

# output population
Output_Pop = ann.Population(geometry=3, neuron=OutputNeuron, name='Output')

# Projections
CM_GPe = ann.Projection(pre=CM, post=GPe, target='exc')
CM_GPe.connect_one_to_one(weights=parameters['strength_efference_copy'])

CM_M1 = ann.Projection(pre=CM, post=M1, target='exc')
CM_M1.connect_one_to_one(weights=1.0)

S1_StrD1 = ann.Projection(pre=S1, post=StrD1, target='mod')
w_s1 = w_2D_to_3D_S1(preDim=parameters['dim_s1'], postDim=StrD1.geometry)
S1_StrD1.connect_from_matrix(w_s1)

PM_StrD1 = ann.Projection(pre=PM, post=StrD1, target='exc', synapse=PostCovarianceNoThreshold, name='PM_D1')
PM_StrD1.connect_all_to_all(0.0)

StrD1_SNr = ann.Projection(pre=StrD1, post=SNr, target='inh', synapse=PreCovariance_inhibitory, name='D1_SNr')
StrD1_SNr.connect_all_to_all(0.0)

GPe_SNr = ann.Projection(pre=GPe, post=SNr, target='inh', name='GPe_SNr')
GPe_SNr.connect_one_to_one(weights=1.0)

# dopa connections
SNc_SNr = ann.Projection(pre=SNc, post=SNr, target='dopa')
SNc_SNr.connect_all_to_all(1.0)

SNc_StrD1 = ann.Projection(pre=SNc, post=StrD1, target='dopa')
SNc_StrD1.connect_all_to_all(1.0)

SNr_VL = ann.Projection(pre=SNr, post=VL, target='inh')
SNr_VL.connect_one_to_one(1.0)

VL_M1 = ann.Projection(pre=VL, post=M1, target='exc')
w_vl_m1 = connect_gaussian_circle(Dim=parameters['dim_bg'], scale=parameters['sig_m1'],
                                  sd=parameters['sig_vl_m1'], A=parameters['A_vl_m1'])
VL_M1.connect_from_matrix(w_vl_m1)

# Output projection
PopCode_out = ann.Projection(pre=M1, post=Output_Pop, target='exc')
w_out = pop_code_output(parameters['motor_orientations'])
PopCode_out.connect_from_matrix(w_out)

# normalize PopCode
PopCode_norm = ann.Projection(pre=M1, post=Output_Pop[0], target='norm')
PopCode_norm.connect_all_to_all(1.0)

# Feedback connection
M1_StrD1 = ann.Projection(pre=M1, post=StrD1, target='exc', synapse=LearningMT)
M1_StrD1.connect_all_to_all(ann.Uniform(min=0.0, max=0.5))

# Reward prediction
StrD1_SNc = ann.Projection(pre=StrD1, post=SNc, target='inh', synapse=DAPrediction)
StrD1_SNc.connect_all_to_all(weights=0.0)

# Laterals
SNr_SNr = ann.Projection(pre=SNr, post=SNr, target='exc', synapse=ReversedSynapse)
SNr_SNr.connect_all_to_all(0.1)

StrD1_StrD1 = ann.Projection(pre=StrD1, post=StrD1, target='inh')
wD1_D1 = laterals_layerwise(Dim=StrD1.geometry, axis=2, weight=0.25)
StrD1_StrD1.connect_from_matrix(wD1_D1)

M1_M1 = ann.Projection(pre=M1, post=M1, target='inh')
M1_M1.connect_all_to_all(0.1)
