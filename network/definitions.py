import ANNarchy as ann

baseline_dopa = ann.Constant('baseline_dopa', 0.1)

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
        r = mp : min = 0.0
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
        firing = 0 : population, bool
        factor_inh = 10.0 : population
    """,
    equations="""
        s_inh = sum(inh)
        aux =   if firing: 
                    firing*(pos(1.0-baseline_dopa-s_inh) + baseline_dopa) + (1-firing)*(-factor_inh*sum(inh))  
                else: baseline_dopa
        tau*dmp/dt + mp =  aux
        r = mp : min = 0.0
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
        tau = 1000.0 : projection
        tau_alpha = 10.0 : projection
        regularization_threshold = 1.0 : projection
        K_burst = 1.0 : projection
        K_dip = 0.4 : projection
        DA_type = 1 : projection
        threshold_pre = 0.0 : projection
        threshold_post = 0.05 : projection
    """,
    equations="""
        tau_alpha*dalpha/dt  + alpha = pos(post.mp - regularization_threshold)
        dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa)
        trace = pos(post.r -  mean(post.r) - threshold_post) * (pre.r - mean(pre.r) - threshold_pre)
        condition_0 = if (trace>0.0) and (w >0.0): 1 else: 0
        dopa_mod =  if (DA_type*dopa_sum>0): DA_type*K_burst*dopa_sum
                    else: condition_0*DA_type*K_dip*dopa_sum
        delta = (dopa_mod* trace - alpha*pos(post.r - mean(post.r) - threshold_post)*pos(post.r - mean(post.r) - threshold_post))
        tau*dw/dt = delta : min = 0.0
    """
)


# Inhibitory synapses STRD1 -> SNr
PreCovariance_inhibitory = ann.Synapse(
    parameters="""
        tau=1000.0 : projection
        tau_alpha=10.0 : projection
        regularization_threshold = 1.0 : projection
        K_burst = 1.0 : projection
        K_dip = 0.4 : projection
        DA_type = 1 : projection
        threshold_pre = 0.05 : projection
        threshold_post = 0.0 : projection
        negterm = 1 : projection
    """,
    equations="""
        tau_alpha*dalpha/dt = pos( -post.mp - regularization_threshold) - alpha
        dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa)
        trace = pos(pre.r - mean(pre.r) - threshold_pre) * (mean(post.r) - post.r  - threshold_post)
        aux = if (trace>0): negterm else: 0
        dopa_mod = if (DA_type*dopa_sum>0): DA_type*K_burst*dopa_sum else: aux*DA_type*K_dip*dopa_sum
        delta = dopa_mod * trace - alpha * pos(trace)
        tau*dw/dt = delta : min = 0.0
    """
)

DAPrediction = ann.Synapse(
    parameters="""
        tau = 10000.0 : projection
   """,
   equations="""
       aux = if (post.mp>0): 1.0 else: 3.0
       delta = aux*pos(post.r - baseline_dopa)*pos(pre.r - mean(pre.r))
       tau*dw/dt = delta : min = 0.0
   """
)