from .params import parameter_1D

from .connections import *
from .definitions import *

# input populations
PM = ann.Population(geometry=parameter_1D['dim_pm'], neuron=BaselineNeuron, name='PM')
S1 = ann.Population(geometry=parameter_1D['dim_s1'], neuron=BaselineNeuron, name='S1')

STN = ann.Population(geometry=parameter_1D['dim_motor'], neuron=BaselineNeuron, name='STN')

# CBGT Loop
StrD1 = ann.Population(geometry=parameter_1D['dim_str'], neuron=LinearNeuron, name='StrD1')
StrD1.noise = 0.05

SNr = ann.Population(geometry=parameter_1D['dim_bg'], neuron=LinearNeuron, name='SNr')
SNr.noise = 0.02
SNr.baseline = 0.6

VL = ann.Population(geometry=parameter_1D['dim_bg'], neuron=LinearNeuron, name='VL')
VL.noise = 0.05
VL.baseline = 0.6

M1 = ann.Population(geometry=parameter_1D['dim_bg'], neuron=LinearNeuron, name='M1')
M1.noise = 0.01
M1.baseline = 0.0

# output population
OutPop = ann.Population(geometry=parameter_1D['dim_motor'], neuron=OutputNeuron, name='Output')

# Projections
S1_StrD1 = ann.Projection(pre=S1, post=StrD1, target='mod')
w_s1 = column_wise_connection(preDim=S1.geometry, postDim=StrD1.geometry)
S1_StrD1.connect_from_matrix(w_s1)

PM_StrD1 = ann.Projection(pre=PM, post=StrD1, target='exc', synapse=PostCovarianceNoThreshold)
PM_StrD1.connect_all_to_all(0.0)

StrD1_StrD1 = ann.Projection(pre=StrD1, post=StrD1, target='inh')
StrD1_StrD1.connect_all_to_all(0.1)

StrD1_SNr = ann.Projection(pre=StrD1, post=SNr, target='inh', synapse=PreCovariance_inhibitory)
w_StrD1_SNr = w_ones_to_all(preDim=StrD1.geometry, postDim=SNr.geometry, weight=0.0)
StrD1_SNr.connect_from_matrix(w_StrD1_SNr)

STN_SNr = ann.Projection(pre=STN, post=SNr, target='exc')
w_stn = row_wise_connection(preDim=STN.geometry, postDim=SNr.geometry)
STN_SNr.connect_from_matrix(w_stn)

# SNr_SNr = ann.Projection(pre=SNr, post=SNr, target='exc', synapse=ReversedSynapse)
# SNr_SNr.connect_all_to_all(0.1)

SNr_VL = ann.Projection(pre=SNr, post=VL, target='inh')
SNr_VL.connect_one_to_one(1.0)

VL_M1 = ann.Projection(pre=VL, post=M1, target='exc')
VL_M1.connect_one_to_one(1.0)

M1_Output = ann.Projection(pre=M1, post=OutPop, target='exc')
w_pool = w_pooling(preDim=M1.geometry)
M1_Output.connect_from_matrix(w_pool)

# Feedback connection
M1_StrD1 = ann.Projection(pre=M1, post=StrD1, target='exc')
M1_StrD1.connect_all_to_all(ann.Uniform(min=0.0, max=0.5))
