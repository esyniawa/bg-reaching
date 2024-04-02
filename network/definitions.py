import ANNarchy as ann

# Neuron definitions
TargetNeuron = ann.Neuron(
    parameters="""
    baseline = 0.0
    phi = 0.0 : population
    """,
    equations="""
    r = baseline + phi * Uniform(-1.0,1.0)
    """
)

OutputNeuron = ann.Neuron(
    parameters="""
    baseline = 0.0
    phi = 0.0 : population
    """,
    equations="""
    r = sum(exc) + baseline + phi * Uniform(-1.0,1.0)
    """
)

BaselineNeuron = ann.Neuron(
    parameters="""
        tau_up = 5.0 : population
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
        baseline = 0.0 : population
        noise = 0.0 : population
        lesion = 1.0 : population
        alpha = 0.8 : population
    """,
    equations="""
        tau*dmp/dt + mp = sum(exc) - sum(inh) + noise*Uniform(-1.0,1.0)
        r = lesion*mp + baseline : min=0.0 
    """
)

StriatumD1Neuron = ann.Neuron(
    parameters="""
        tau = 10.0 : population
        baseline = 0.0 : population
        noise = 0.0 : population
        lesion = 1.0 : population
        alpha = 0.8 : population
    """,
    equations="""
        tau*dmp/dt + mp = sum(mod) * (sum(exc) - sum(inh)) + noise*Uniform(-1.0,1.0)
        r = lesion*mp + baseline: min = 0.0

        r_mean = alpha * r_mean + (1 - alpha) * r
    """
)

DopamineNeuron = ann.Neuron(
    parameters="""
        tau = 10.0 : population
        firing = 0 : population
        inhibition = 0.0 : population
        baseline = 0.0 : population
        exc_threshold = 0.0 : population
        factor_inh = 10.0 : population
    """,
    equations="""
        ex_in = if (sum(exc)>exc_threshold): 1 else: 0
        s_inh = sum(inh)
        aux =   if (firing>0): 
                    (ex_in)*(pos(1.0-baseline-s_inh) + baseline) + (1-ex_in)*(-factor_inh*sum(inh))  
                else: baseline
        tau*dmp/dt + mp =  aux
        r = if (mp>0.0): mp else: 0.0
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
        tau=50.0 : projection
        tau_alpha=10.0 : projection
        regularization_threshold=1.0 : projection
        baseline_dopa = 0.1 : projection
        K_burst = 1.0 : projection
        K_dip = 0.4 : projection
        DA_type = 1 : projection
        threshold_pre=0.0 : projection
        threshold_post=0.0 : projection
    """,
    equations="""
        tau_alpha*dalpha/dt  + alpha = pos(post.mp - regularization_threshold)
        trace = pos(post.r -  mean(post.r) - threshold_post) * pos(pre.r - threshold_pre)
        tau * dweight/dt = trace - alpha*pos(post.r - mean(post.r) - threshold_post)
        w = weight : min=0
    """
)

# Inhibitory synapses STRD1 -> SNr and STRD2 -> GPe
PreCovariance_inhibitory = ann.Synapse(
    parameters="""
    tau = 1000.0 : projection
    DA_type = 1 : projection
    threshold_pre = 0.0 : projection
    threshold_post = 0.0 : projection
    """,
    equations="""
        trace = pos(pre.r - threshold_pre) * (mean(post.r) - post.r - threshold_post)
        tau * dweight/dt = DA_type * trace
        w = weight : min=0
    """
)

DAPrediction = ann.Synapse(
    parameters="""
        tau = 100000.0 : projection
        baseline_dopa = 0.1 : projection
   """,
   equations="""
       aux = if (post.sum(exc)>0): 1.0 else: 3.0
       delta = aux*pos(post.r - baseline_dopa)*pos(pre.r - mean(pre.r))
       tau*dw/dt = delta : min=0
   """
)

