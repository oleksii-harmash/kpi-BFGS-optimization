# BFGS method parameters
e_BFGS = 10e-3         # convergence accuracy (epsilon-neighbourhood)
max_BFGS = 600         # max available iterations
method = 'dsk_powell'   # 'dsk_powell' or 'golden_section'
criterion = 'norm'      # stopping criterion: 'delta' or 'norm'
h = 0.0001              # value of step in derivative formula
schema = 'central'      # derivative schema: 'central', 'right', 'left'

# Svenn's algorithm parameters
init_lambda = 0.12         # initial value of lambda
delta_cf = 0.08        # value of coefficient in delta lambda

# Golden section method
e_GS = 0.01            # golden section convergence accuracy

# DSK-Powell method
e_DSK = 0.01
