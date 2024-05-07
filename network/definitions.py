import ANNarchy as ann

# Neuron definitions
PoolingNeuron = ann.Neuron(
    parameters="""
        r_scale = 1.0 : population
    """,
    equations="""
        r = r_scale * sum(exc)
    """
)

OutputNeuron = ann.Neuron(
    equations="""
        r = if sum(norm) > 0.0: sum(exc) / sum(norm) else: sum(exc)
    """
)

BaselineNeuron = ann.Neuron(
    parameters="""
        tau_up = 10.0 : population
        tau_down = 20.0 : population
        baseline = 0.0
        noise = 0.0 : population
    """,
    equations="""
        base = baseline + noise * Uniform(-1.0,1.0): min=0.0
        dr/dt = if (baseline>0.01): (base-r)/tau_up else: -r/tau_down : min=0.0
    """,
    name="Baseline Neuron",
    description="Time-dynamic neuron with baseline to be set. "
)

LinearNeuron = ann.Neuron(
    parameters="""
        tau = 10.0 : population
        baseline = 0.0
        noise = 0.0 : population
    """,
    equations="""
        tau*dmp/dt + mp = sum(exc) - sum(inh) + noise*Uniform(-1.0,1.0) + baseline
        r = mp : min=0.0 
    """
)

StriatumD1Neuron = ann.Neuron(
    parameters="""
        tau = 10.0 : population
        baseline = 0.0 : population
        noise = 0.0 : population
        alpha = 0.8 : population
    """,
    equations="""
        tau*dmp/dt + mp = sum(mod) * (sum(exc) - sum(inh)) + noise*Uniform(-1.0,1.0) + baseline
        r = tanh(mp): min = 0.0
        
        r_mean = alpha * r_mean + (1 - alpha) * r
    """
)

SNrNeuron = ann.Neuron(
    parameters="""
        tau = 10.0 : population
        baseline = 0.0 : population
        noise = 0.0 : population
        alpha = 0.8 : population
    """,
    equations="""
        tau*dmp/dt + mp = sum(exc) - sum(inh) + noise*Uniform(-1.0,1.0) + baseline
        r = mp : min = 0.0
        
        r_mean = alpha * r_mean + (1 - alpha) * r
    """
)

DopamineNeuron = ann.Neuron(
    parameters="""
        tau = 10.0 : population
        baseline = 0.1 : population
        firing = 0 : population, bool
    """,
    equations="""
        aux =   if (firing>0): (1. - baseline)  
                else: 0.0
        tau*dmp/dt + mp =  aux
        r = mp: min=0.0
    """
)

# Synapse definitions
ReversedSynapse = ann.Synapse(
    parameters="""
        reversal = 1.2 : projection
    """,
    psp="""
        w*pos(reversal-pre.r)
    """,
    name="Reversed Synapse",
    description="Higher pre-synaptic activity lowers the synaptic transmission and vice versa."
)

# DA_typ = 1  ==> D1 type  DA_typ = -1 ==> D2 type
PostCovarianceNoThreshold = ann.Synapse(
    parameters="""
        tau=20.0 : projection
        tau_alpha=10.0 : projection
        regularization_threshold=0.5 : projection
        DA_type=1 : projection
        threshold_pre=0.05 : projection
        threshold_post=0.05 : projection
        eta=1.0 : projection
    """,
    equations="""
        tau_alpha*dalpha/dt  + alpha = pos(post.r - regularization_threshold)
        dopa_mod = post.sum(dopa)
        trace = eta * pos(post.r - post.r_mean - threshold_post) * pos(pre.r - threshold_pre)
        tau * dweight/dt = dopa_mod * trace 
        w = dopa_mod * (weight - alpha) : min = 0.0
    """
)

# Inhibitory synapses STRD1 -> SNr
PreCovariance_inhibitory = ann.Synapse(
    parameters="""
        tau = 20.0 : projection
        tau_alpha=10.0 : projection
        DA_type = 1 : projection
        threshold_pre = 0.01 : projection
        threshold_post = 0.01 : projection
        regularization_threshold = 0.6 : projection
        eta = 1.0 : projection
    """,
    equations="""
        tau_alpha * dalpha/dt + alpha = pos(-post.r + regularization_threshold)
        trace = eta * pos(pre.r - post.r_mean - threshold_pre) * (post.r_mean - post.r - threshold_post)
        dopa_mod = post.sum(dopa)
        tau * dweight/dt = dopa_mod * trace
        w = dopa_mod * (weight) : min = 0.0
    """
)

# GPe_Synapse = ann.Synapse(
#     parameters="""
#         weight = 1.0 : projection
#     """,
#     equations="""
#         dopa_mod = post.sum(dopa)
#         w = dopa_mod * weight: min = 0.0
#     """
# )

DAPrediction = ann.Synapse(
    parameters="""
        tau = 100000.0 : projection
        baseline_dopa = 0.1 : projection
    """,
    equations="""
        aux = if (post.sum(exc)>0): 1.0 else: 3.0
        delta = aux*pos(post.r - baseline_dopa)*pos(pre.r - mean(pre.r))
        tau*dw/dt = delta : min = 0.0
   """
)
