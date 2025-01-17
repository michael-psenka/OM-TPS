import sympy as sp
import numpy as np
import torch

from torch.func import vjp as vjpfunc
from torch.func import vmap


# Define the symbol
x1, x2, x3, x4, x5 = sp.symbols("x1 x2 x3 x4 x5")

# Create a vector of functions
functions = [
    sp.sin(x1) + x3,  # 1st function: sine
    sp.cos(x2),  # 2nd function: cosine
    sp.exp(x3) + x1 * sp.log(x2 + 2) + x5**3,  # 3rd function: exponential
    sp.log(x4 + 1)
    + sp.sin(x1),  # 4th function: natural logarithm (shifted by 1 for domain)
    x5**2,  # 5th function: quadratic
]

torch_transform = lambda x: torch.stack(
    [
        torch.sin(x[0]) + x[2],  # 1st function: sine
        torch.cos(x[1]),  # 2nd function: cosine
        torch.exp(x[2])
        + x[0] * torch.log(x[1] + 2)
        + x[4] ** 3,  # 3rd function: exponential
        torch.log(x[3] + 1)
        + torch.sin(x[0]),  # 4th function: natural logarithm (shifted by 1 for domain)
        x[4] ** 2,
    ]
)  # 5th function: quadratic

# Create a vector from these functions
vector_f = sp.Matrix(functions)

# Calculate the Jacobian matrix
jacobian_f = vector_f.jacobian([x1, x2, x3, x4, x5])

# Create a vector from these functions
vector_f = sp.Matrix(functions)

# Calculate the Jacobian matrix
jacobian_f = vector_f.jacobian([x1, x2, x3, x4, x5])

batches = 10
input_arr = []
grad_arr = []
trace_arr = []

for _ in range(batches):

    rx = np.random.rand(5)

    # Substitute these random values into the Jacobian matrix
    jacobian_evaluated = jacobian_f.subs(
        {x1: rx[0], x2: rx[1], x3: rx[2], x4: rx[3], x5: rx[4]}
    )

    # Output the evaluated Jacobian matrix
    # print("Evaluated Jacobian")
    # sp.pprint(jacobian_evaluated)

    diagonal_elements = [
        jacobian_evaluated[i, i] for i in range(jacobian_evaluated.shape[0])
    ]
    # print("Diagonal elements")
    # sp.pprint(diagonal_elements)
    # print("Sum")
    # sp.pprint(sum(diagonal_elements))
    trace_arr.append(sum(diagonal_elements))

    # Batched test

    trx = torch.tensor(rx, requires_grad=True, dtype=torch.float32)

    input_arr.append(trx)


print(trace_arr)


torch_inputs = torch.stack(input_arr, axis=0)
# grads = torch_transform(torch_inputs)
print("Input shape: ", torch_inputs.shape)


def hutchinson_trace_approximation(x_inputs, runfor=100):
    approxs = []
    for i in range(runfor):
        rand = torch.normal(
            torch.zeros(
                5,
            ),
            1,
        )

        grads, Gr_func = vjpfunc(torch_transform, x_inputs)
        (Gr,) = Gr_func(rand)
        approxs.append(
            torch.sum(
                Gr * rand,
            )
        )

    return sum(approxs) / len(approxs)


torch_res = vmap(hutchinson_trace_approximation, randomness="different")(torch_inputs)
print("Accurate traces", trace_arr)
print("Torch traces: ", torch_res)
