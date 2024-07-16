A constrained Variational Quantum Eigensolver algorithm to find the minimum cut of a weighted graph as a partition.

Initially, adapted a Variational Quantum Deflation algorithm but optimisation was stuck on barren plateaus for trivial cases. 
This used a QAOA ansatz.

Instead, insisting source and sink nodes in different partitions and applying Twolocal entanglement layers on remaining qubits works.
Further investigation using Nlocal could be done.

Main Packages: 
- Qiskit v1.0.2
- Scipy v1.13.1
