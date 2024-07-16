'''
VQE alg applied to find min cut (= max flow) of a graph
 
Aims: 
- test various circuit depths in twolocal and how it affects speed & accuracy
- test various transpilation parameters, simulators etc
- different minimisation methods eg grad free (COBYLA)

Testing:
- generate random graphs with rustworkx, first try <=5 nodes and then <= 10
 
'''

import matplotlib.pyplot as plt
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, SparsePauliOp
#from qiskit.circuit.library import QAOAAnsatz as  qa 

# rustworkx graph to (weighted) Pauli string
'''
random graph generator w rustworkx:

rustworkx.undirected_gnp_random_graph(...[, seed])
G_np, n is nodes and p is probability an edge is included

rustworkx.has_path(graph, source, target, as_undirected=False) # checks if path exists
PyGraph.extend_from_weighted_edge_list(edge_list, /) # can add weights (create random list) to edges
PyGraph.edge_list() # returns list of edges (without weights)
rustworkx.visualization.mpl_draw(graph) # draw graph w. Matplotlib

'''

import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
import numpy as np

edge_list = [(0, 1, 1.0), (0, 2, 1.0), (0, 4, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)]
#edge_list = [(0, 1, 1.0),(0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (1, 3, 2.0)]

source_sink = [0,2]
num_nodes = max([edge[1] for edge in edge_list])+1

def edge_to_ham(edge_list, num_nodes):
    '''
    Parameters:
    edge_list: list type for graph
    num_nodes: int type inferred from graph at earlier stage
    source_sink: [source, sink] as int types " "
    '''
    pauli_ops = []
    n = num_nodes
    for edge in edge_list:
        edge_op = 'I'*(n-edge[1]-1)+'Z'+'I'*(edge[1]-edge[0]-1)+'Z'+'I'*edge[0]
        pauli_ops.append((edge_op,-1*edge[2])) # need -1 for finding min cut
    cost_ham = SparsePauliOp.from_list(pauli_ops)
    return cost_ham

# ansatz circuit, QAOAAnsatz

cost_ham = edge_to_ham(edge_list,num_nodes)
#ansatz = qa(cost_operator = cost_ham, reps =5)# this doesn't include VQD term

from qiskit.circuit.library import TwoLocal
#variational form doesnt act on source/sink qubits
variational_form = TwoLocal(
    num_nodes-2,
    rotation_blocks=["rz", "ry"],
    entanglement_blocks="cx",
    entanglement="linear",
    reps=2,
)
qc = QuantumCircuit(2)
qc.x(1) #requirement source and sink are in different partitions
ansatz = QuantumCircuit(num_nodes)
ansatz.compose(qc, source_sink, inplace=True)
ansatz.compose(variational_form, [i for i in range(num_nodes) if i not in source_sink], inplace=True)

'''
could frame problem as adding to cost function a larger term if source/sink are in same partition
however, no. qubits reduced if have ansatz constrained initially so can increase number of entanglement layers here
and seem to be stuck on a barren plateau when executing this

def source_sink_circuit(source_sink,num_nodes):
    # equal superpos state of all basis states with source bit = sink bit
    qc = QuantumCircuit(num_nodes)
    source = source_sink[0]
    sink = source_sink[1]
    for i in range(num_nodes):
        if i != source:
            qc.h(i)
    qc.cx(sink,source)#conditioned on sink
    return qc

qc = source_sink_circuit(source_sink,num_nodes)
ansatz2 = ansatz1.compose(qc.inverse()) # VQD term to ensure orthogonality
'''
# minimisation routine with cost function
def cost_func(params, ansatz, hamiltonian, estimator):
    pub = (ansatz,hamiltonian,params)
    cost = estimator.run([pub]).result()[0].data.evs
    return cost

import time
from scipy.optimize import minimize
from scipy.optimize._optimize import OptimizeResult
from qiskit.primitives import StatevectorEstimator, StatevectorSampler, PrimitiveJob
# basic simulation 
estimator = StatevectorEstimator()
sampler = StatevectorSampler()

# noisy simulation/ real hardware
'''
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_aer import AerSimulator
'''

# random initial parameters
import numpy as np
start_time = time.time()
x0 = 2*np.pi*np.random.rand(ansatz.num_parameters)
result = minimize(cost_func, x0, args=(ansatz,cost_ham,estimator))
# output minimal solution
print("--- %s seconds ---" % (time.time() - start_time))
print(result)

opt_params = result.x #result of minimisation

# determine partition & max flow
#
from qiskit.visualization import plot_distribution
qc = ansatz.assign_parameters(opt_params) #optimal param quantum circuit
#add measurements to ansatz circuit
qc.measure_all()

#run circuit
job = sampler.run([qc],shots=1024)
data_pub = job.result()[0].data
counts = data_pub.meas.get_counts()
#hist = plot_histogram(counts,sort='value_desc')
#plot_distribution(counts)
#plt.show()
#print(counts)
binary_string = max(counts.items(),key=lambda k: k[1])[0]
lst = np.asarray([int(y) for y in reversed(list(binary_string))])
colors = ["r" if lst[i] == 0 else "c" for i in range(num_nodes)]
G = rx.PyGraph()
G.add_nodes_from(np.arange(0, num_nodes, 1))
G.add_edges_from(edge_list)
draw_graph(G, pos=rx.shell_layout(G), with_labels=True, edge_labels=str, node_color=colors)
plt.show()
