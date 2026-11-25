# Q&A preparation (3 minutes after the live pitch)

Answer in two or three sentences, then stop. If you do not know, say so and say how you would find out.

**Q: Simulated annealing solves this faster. Why use quantum at all?**
At 6 to 12 nodes, yes, and I show that in my own table. My goal this weekend was to build the full
pipeline, from microgrid to QUBO to QAOA to hardware, with honest baselines, and to measure where it breaks:
noise and circuit depth. The interesting regime is large sparse grids and multi-hour planning, where exact
methods stop scaling. That is my next step.

**Q: How does this scale?**
One qubit per node per time step. Couplings come only from sources that can reach a common node, so on a
sparse feeder each qubit has a bounded number of neighbours, and depth grows with grid density, not with
`n²`. My depth-vs-density plot shows this. The limit today is two-qubit depth under noise: my 12-qubit,
3-layer circuits already reach about 230 two-qubit layers after routing on Heron, and that is where
noise visibly erodes the gain.

**Q: What did noise do to your results?**
The noise model keeps the same parameters working, but it reduces the probability of the optimal plan from
38.5 to 13.9 times random at 12 qubits, so 95% confidence needs about 1,400 shots instead of 425.
Because I keep the best sample, the optimal dispatch was still found in all 36 instances. On hardware: **[result and job IDs]**.

**Q: How did you choose lambda, the penalty weight?**
From economics, not tuning. lambda is set so that, at the margin, a mismatch of half a typical node's demand
costs the value of lost load, 2 USD per kWh (an assumption I state). A 15% operating reserve shifts the
target so the optimum prefers keeping the lights on.

**Q: Why is % demand met not 100%?**
With binary on/off and one-hop sharing, exact balance at every node is often impossible. The optimum trades
small shortfalls against curtailment. Greedy switches on more capacity yet meets slightly less demand
(92.8% against 93.2%), because the extra supply lands where nobody needs it. A real deployment would add battery state of charge and partial dispatch levels.

**Q: Where does the data come from?**
It is synthetic and seeded, following the hackathon's data rule. The value ranges are stated assumptions
in `instance.py`. Next, I would calibrate them with a partner microgrid's load and solar profiles.

**Q: Why QAOA and not VQE, or a quantum annealer?**
Dispatch is a combinatorial problem with a diagonal cost, which is what QAOA is built for, and it runs on
IBM's gate-based hardware. I added CVaR, which focuses the optimiser on the best samples, and INTERP warm
starts, which make deeper circuits trainable with few evaluations.

**Q: What error mitigation did you use?**
Dynamical decoupling and Pauli twirling on the sampler, and zero-noise extrapolation on the estimator for the
energy. On the local noise model, suppression does not help, because that noise is already stochastic. Real
devices have coherent errors, which is what it targets.

**Q: Who would actually use this, and how?**
A microgrid cooperative or a rural utility, through the planner running every hour on a laptop or a cloud
simulator today. Quantum hardware comes in as problem sizes grow. The output is a simple on/off schedule they
already understand.

**Q: What would you do with another month?**
Multi-hour dispatch with batteries, 50 to 100 node sparse grids on Heron compared with a MILP solver, and real
data from a partner site.
