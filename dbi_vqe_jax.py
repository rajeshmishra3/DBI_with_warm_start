import quimb as qu
import quimb.tensor as qtn
import scipy
import numpy as np
import tqdm
import jax
from jax import numpy as jnp
from functools import reduce
import flax.linen as nn
import optax
import matplotlib.pyplot as plt

### TODO
# 1. Rewrite the optimization using jax (delve a little into jax for curiosity). 
# 2. Try a few different vqe ansatz.
# 3. Finish the DBI part (understand the theory behind it).
# 4. Try larger size chains.

def build_xxz_hamiltonian(n_sites, couplings):
    """
    Builds the XXZ Hamiltonian with transverse field.

    Input
    n_sites: number of sites or number of qubits
    couplings: coupling in the 3 directions and the transverse field strength

    Output
    Hamiltonian as an MPO
    Hamiltonian as Pauli decomposition
    """

    Jx, Jy, Jz, h = couplings
    paulis = [] 
    coeffs = [] 
    sites = []

    mpo_terms = []

    for i in range(n_sites -1):
        if abs(Jx)>0:
            mpo_terms.append((4*Jx, i, i + 1, 'X', 'X'))
            paulis.append('XX')
            coeffs.append(Jx)
            sites.append([i,i+1]) 
        if abs(Jy)>0:
            mpo_terms.append((4*Jy, i, i + 1, 'Y', 'Y'))
            paulis.append('YY')
            coeffs.append(Jy)
            sites.append([i,i+1]) 
        if abs(Jz)>0:
            mpo_terms.append((4*Jz, i, i + 1, 'Z', 'Z'))
            paulis.append('ZZ')
            coeffs.append(Jz)
            sites.append([i,i+1]) 

    # Add transverse field term
    # for i in range(n_sites):
    #     mpo_terms.append((2*h, i, 'Z'))
    #     paulis.append('Z')
    #     sites.append([i])
    #     coeffs.append(h)
        
    H = [coeffs,paulis,sites]

    # Build the MPO
    mpo = qtn.SpinHam1D(S=0.5)
    for term in mpo_terms:
        if len(term) == 5:
            coeff, site1, site2, op1, op2 = term
            mpo[site1,site2] += coeff, op1, op2
        elif len(term) == 3:
            coeff, site, op = term
            mpo[site] += coeff, op
    
    return mpo.build_mpo(n_sites), H

def VQE(params):
    """
    Build a tensor network quantum circuit ansatz.

    Input
    params: angles for the rotation gates

    Output
    circuit: circuit in MPS form with a chosen max bond dimension
    """
    circuit =  qtn.CircuitMPS(n_sites, max_bond=128,cutoff=10**-6)
    # Apply parameterized rotations and interactions
    idx = 0 
    for _ in range(layers):
        for i in range(n_sites):
            circuit.apply_gate('RY', params[idx], i)
            idx +=1
            circuit.apply_gate('RZ', params[idx], i)
            idx +=1

        for i in range(0,n_sites - 1,2):
            circuit.apply_gate('CZ',i, i + 1)
            
        for i in range(1,n_sites - 1,2):
            circuit.apply_gate('CZ', i, i + 1)

    return circuit

def multi_pauli(op_string):
    """
    Builds a tensor product of the Pauli operator.
    """
    if len(op_string)==1:
        return qu.pauli(op_string)
    # Generate the Kronecker product for the Pauli string
    paulis = {'I': qu.eye(2), 'X': qu.pauli('X'), 'Y': qu.pauli('Y'), 'Z': qu.pauli('Z')}
    return reduce(qu.kron, (paulis[op] for op in op_string))
    
def evaluate(circuit,H):
    """
    Computes the energy of the system.

    Input 
    circuit: |psi>
    H: Hamiltonian

    Output
    Energy: <psi|H|psi>
    """
    coeffs, paulis, sites = H 
    energy = 0 
    for c,p,s in zip(coeffs, paulis, sites):     
        energy += c*circuit.local_expectation(G=multi_pauli(p), where=s, normalized=True).real
 
    return energy 
    
def cost_function(params):
    """
    Cost function computation

    Input
    params: parameters of the PQC (ansatz)
    H: Hamiltonian

    Output:
    Cost/Energy
    """
    
    # circuit = qtn.CircuitMPS(psi0=params)
    circuit = VQE(params)
    
    return evaluate(circuit,H)

class CustomModule(nn.Module):

    def setup(self):
        # strip out the initial raw arrays
        # params, skeleton = qtn.pack(psi)
        params = initial_params
        # save the stripped 'skeleton' tn for use later
        # self.skeleton = skeleton

        # assign each array as a parameter to optimize
        # self.params = {
        #     i: self.param(f'param_{i}', lambda _: data)
        #     for i, data in params.items()
        # }
        self.params = {
            i: self.param(f'param_{i}', lambda _: params[i])
            for i in range(len(params))
        }

    def __call__(self):
        # psi = qtn.unpack(self.params, self.skeleton)
        return cost_function(jnp.array(list(self.params.values())))
    
def append_circuits(circuit1, circuit2):
    # Append all gates from circuit2 to circuit1
    for gate in circuit2.gates:
        circuit1.apply_gate(gate)
    return circuit1

def invert_circuit(circuit):
   
    inverse_circuit = qtn.Circuit(n_sites)
    
    # Reverse the gate application
    for gate in reversed(circuit.gates):
     
        label = gate.label
        qubits = gate.qubits
      
        params = [-p for p in gate.params] if gate.params else None  # Negate parameters for rotation gates
    
        inverse_circuit.apply_gate(label, params,*qubits)

    return inverse_circuit

def D(circuit,params):
    idx = 0 
    for i in range(n_sites):
        circuit.apply_gate('RZ', 2*params[idx], i)
        idx +=1

    for i in range(0,n_sites - 1,2):
        circuit.apply_gate('RZZ', 2*params[idx],i, i + 1)
        idx +=1
            
    for i in range(1,n_sites - 1,2):
        circuit.apply_gate('RZZ', 2*params[idx],i, i + 1)
        idx +=1

    return circuit 

def Trotter(circuit, couplings, t, nsteps=2):
    # To do: implement second order  
    
    Jx, Jy, Jz, h = couplings
    dt = t / nsteps  # time step for each Trotter step
    
    # Apply Trotter steps
    for _ in range(nsteps):
        # Apply magnetic field (RZ gates for each site)
        for i in range(n_sites):
            if abs(h) > 0:
                circuit.apply_gate('RZ', 2 * h * dt, i)
        
        # Apply interaction terms (RXX, RYY, RZZ gates for coupling terms)
        for i in range(0, n_sites - 1, 2):  # even-indexed sites
            if abs(Jz) > 0:
                circuit.apply_gate('RZZ', 2 * Jz * dt, i, i + 1)
            if abs(Jx) > 0:
                circuit.apply_gate('RXX', 2 * Jx * dt, i, i + 1)
            if abs(Jy) > 0:
                circuit.apply_gate('RYY', 2 * Jy * dt, i, i + 1)
        
        for i in range(1, n_sites - 1, 2):  # odd-indexed sites
            if abs(Jz) > 0:
                circuit.apply_gate('RZZ', 2 * Jz * dt, i, i + 1)
            if abs(Jx) > 0:
                circuit.apply_gate('RXX', 2 * Jx * dt, i, i + 1)
            if abs(Jy) > 0:
                circuit.apply_gate('RYY', 2 * Jy * dt, i, i + 1)
    
    return circuit


def dbi(s,vqe_params):
    circuit = qtn.CircuitMPS(n_sites, max_bond=1024,cutoff=10**-8)
    
    circuit_vqe = VQE(vqe_params) 
    circuit_vqe_inverse = invert_circuit(circuit_vqe)


    params_d = np.random.uniform(0., 2*np.pi, 2*n_sites-1)

    circuit = D(circuit,params_d)
        
    circuit = append_circuits(circuit,circuit_vqe)
    circuit = Trotter(circuit,couplings,s,nsteps=4) 
    circuit = append_circuits(circuit,circuit_vqe_inverse)

    circuit = D(circuit,-params_d)

    circuit = append_circuits(circuit,circuit_vqe)

    return circuit 

# Define parameters for the XXZ Hamiltonian
n_sites = 50 # Number of sites in the chain
Jx = 1 # Coupling in the x-direction
Jy = 1  # Coupling in the y-direction
Jz = 0.6  # Coupling in the z-direction
h =  +0.4  # Transverse field strength
couplings = np.array([Jx,Jy,Jz,h])/n_sites 

# Build the MPO Hamiltonian
hamiltonian, H = build_xxz_hamiltonian(n_sites, couplings)

# Ground state calculation using VQE 
layers = 2
n_params = 2*layers * (1 * n_sites)
initial_params = np.random.normal(0,10,size=n_params)
# psi = VQE(initial_params).psi

# Optimization using jax, flax and optax
model = CustomModule()
params = model.init(jax.random.PRNGKey(42))
loss_grad_fn = jax.value_and_grad(model.apply)

# tx = optax.adabelief(learning_rate=0.01)
tx = optax.adabelief(learning_rate=0.01)
opt_state = tx.init(params)

# @jax.jit
def step(params, opt_state):
    # our step: compute the loss and gradient, and update the optimizer
    loss, grads = loss_grad_fn(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

its = 100
pbar = tqdm.tqdm(range(its))
list_energy = []

for _ in pbar:
    params, opt_state, loss_val = step(params, opt_state)
    list_energy.append(loss_val)
    pbar.set_description(f"{loss_val}")

param_opt = initial_params.copy()

for i in range(n_params):
    param_opt[i] = params['params'][f'param_{i}'].__array__()

# resinsert the raw, optimized arrays
# for i, t in param_opt.tensor_map.items():
#     t.modify(data=params['params'][f'param_{i}'].__array__())

optimal_energy = cost_function(param_opt)

dbi_best = 0

for s in tqdm.tqdm(np.logspace(-2, 0, 200)):
    psi = dbi(s,param_opt)
    energy = evaluate(psi,H)
    dbi_best = min(dbi_best, energy)

plt.plot(list_energy)
plt.show()

print(optimal_energy, dbi_best)