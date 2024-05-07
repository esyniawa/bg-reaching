from .params import parameter_1D, state_space

from .connections import *
from .definitions import *


# input populations
PM = ann.Population(geometry=state_space.shape[:2], neuron=BaselineNeuron, name='PM')
S1 = ann.Population(geometry=parameter_1D['dim_s1'], neuron=BaselineNeuron, name='S1')

Cortex = ann.Population(geometry=parameter_1D['dim_motor'], neuron=BaselineNeuron, name='M1_ventral')
SNc = ann.Population(geometry=1, neuron=DopamineNeuron, name='SNc')

# transmission populations into putamen
CM = ann.Population(geometry=parameter_1D['dim_motor'], neuron=LinearNeuron, name='CM')
CM.noise = 0.0

# CBGT Loop
StrD1 = ann.Population(geometry=parameter_1D['dim_str'], neuron=StriatumD1Neuron, name='StrD1')
StrD1.noise = 0.0

GPe = ann.Population(geometry=parameter_1D['dim_motor'], neuron=LinearNeuron, name='GPe')
GPe.noise = 0.01
GPe.baseline = 0.2

SNr = ann.Population(geometry=parameter_1D['dim_bg'], neuron=SNrNeuron, name='SNr')
SNr.noise = 0.02
SNr.baseline = 1.0

VL = ann.Population(geometry=parameter_1D['dim_bg'], neuron=LinearNeuron, name='VL')
VL.noise = 0.05
VL.baseline = 0.6

M1 = ann.Population(geometry=parameter_1D['dim_bg'], neuron=LinearNeuron, name='M1')
M1.tau = 20.
M1.noise = 0.02
M1.baseline = 0.0

# output population
Out_PopCode_Pooling = ann.Population(geometry=parameter_1D['dim_motor'], neuron=PoolingNeuron, name='PopCode')
Out = ann.Population(geometry=3, neuron=OutputNeuron, name='Output')

# Projections
Cortex_CM = ann.Projection(pre=Cortex, post=CM, target='exc')
Cortex_CM.connect_one_to_one(weights=0.3)

CM_GPe = ann.Projection(pre=CM, post=GPe, target='exc')
CM_GPe.connect_one_to_one(weights=1.0)

CM_M1 = ann.Projection(pre=CM, post=M1, target='exc')
w_M1 = column_wise_connection(preDim=parameter_1D['dim_motor'], postDim=M1.geometry)
CM_M1.connect_from_matrix(w_M1)

S1_StrD1 = ann.Projection(pre=S1, post=StrD1, target='mod')
w_s1 = row_wise_connection(preDim=parameter_1D['dim_s1'], postDim=StrD1.geometry)
S1_StrD1.connect_from_matrix(w_s1)

PM_StrD1 = ann.Projection(pre=PM, post=StrD1, target='exc', synapse=PostCovarianceNoThreshold, name='PM_D1')
PM_StrD1.connect_all_to_all(0.0)

StrD1_StrD1 = ann.Projection(pre=StrD1, post=StrD1, target='inh')
StrD1_StrD1.connect_all_to_all(0.2)

StrD1_SNr = ann.Projection(pre=StrD1, post=SNr, target='inh', synapse=PreCovariance_inhibitory, name='D1_SNr')
w_StrD1_SNr = w_ones_to_all(preDim=StrD1.geometry, postDim=SNr.geometry, weight=0.0)
StrD1_SNr.connect_from_matrix(w_StrD1_SNr)

GPe_SNr = ann.Projection(pre=GPe, post=SNr, target='inh', name='GPe_SNr', synapse=GPe_Synapse)
w_gpe = column_wise_connection(preDim=parameter_1D['dim_motor'], postDim=SNr.geometry)
GPe_SNr.connect_from_matrix(w_gpe)

SNc_SNr = ann.Projection(pre=SNc, post=SNr, target='dopa')
SNc_SNr.connect_all_to_all(1.0)

SNc_StrD1 = ann.Projection(pre=SNc, post=StrD1, target='dopa')
SNc_StrD1.connect_all_to_all(1.0)

# SNr_SNr = ann.Projection(pre=SNr, post=SNr, target='exc', synapse=ReversedSynapse)
# SNr_SNr.connect_all_to_all(0.1)

SNr_VL = ann.Projection(pre=SNr, post=VL, target='inh')
SNr_VL.connect_one_to_one(1.0)

VL_M1 = ann.Projection(pre=VL, post=M1, target='exc')
VL_M1.connect_one_to_one(1.0)

M1_Output = ann.Projection(pre=M1, post=Out_PopCode_Pooling, target='exc')
w_pool = w_pooling(preDim=M1.geometry)
M1_Output.connect_from_matrix(w_pool)

PopCode_out = ann.Projection(pre=Out_PopCode_Pooling, post=Out, target='exc')
w_out = pop_code_output(parameter_1D['motor_orientations'])
PopCode_out.connect_from_matrix(w_out)

PopCode_norm = ann.Projection(pre=Out_PopCode_Pooling, post=Out[0], target='norm')
PopCode_norm.connect_all_to_all(1.0)

# Feedback connection
M1_StrD1 = ann.Projection(pre=M1, post=StrD1, target='exc')
M1_StrD1.connect_all_to_all(ann.Uniform(min=0.0, max=1.0))
