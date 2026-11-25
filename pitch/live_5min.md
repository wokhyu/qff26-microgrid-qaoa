# Live final: 5-minute pitch and 3-minute Q&A (Team Qilowatt, solo)

Same story as the video, plus a 90-second live demo. Rubric: Technical 30, Impact 25, Innovation and
feasibility 25, Pitch 20. If scores tie, the higher Technical score wins, then Impact. So the technical slide
and the honest comparison carry the most weight.

Presenting alone means you also drive the demo. Keep the slides and the app in two browser tabs, or two
windows you switch with Alt+Tab, and rehearse the switch.

## Run of show

| Time | Slide or screen | Key line |
|---|---|---|
| 0:00–0:30 | 1. Hook: village or island at night | "Every hour, someone running a microgrid decides which resources to switch on. Too little means blackouts. Too much wastes clean energy or burns diesel." |
| 0:30–1:00 | 2. Who benefits: one verified statistic, the operator and the households | "I measure what they care about: demand met, waste, diesel and cost." |
| 1:00–1:45 | 3. Architecture (`docs/architecture.png`) | "One node, one qubit. Cost plus local balance with a 15% reserve becomes a QUBO, then an Ising Hamiltonian, then QAOA. Sparse grids give sparse circuits. I train on a simulator and reuse the parameters on noise and on hardware." |
| 1:45–3:15 | 4. **Live demo** (Streamlit) | See the demo script below |
| 3:15–4:15 | 5. Results: `baselines.png` and the shots table | Greedy finds the optimal plan in 8 of 36 grids and leaves 59% more surplus. QAOA finds it in all 36. At 12 qubits, 95% confidence takes about 12,000 random guesses against 425 QAOA shots, or 1,400 under Heron noise. "Simulated annealing is faster at this size. No speed-up claimed." |
| 4:15–4:45 | 6. Hardware: job IDs, depth vs density | "Real Heron run: **[result]**. Job IDs are in the repo. Two-qubit depth grows with grid density. That is today's limit." |
| 4:45–5:00 | 7. Next steps and the ask | "Multi-hour planning, 50 to 100 node grids, a partner microgrid. Thank you." |

## Demo script (90 seconds)

1. Open the app on its defaults (n = 8, density 0.3, seed 5, p = 2), already cached. Run it once before you go on so the
   result loads instantly.
2. "Hollow circles are off, filled are on. The percentage under each node is the demand served." Show the
   **QAOA (best of shots)** map.
3. Switch to **Greedy merit-order**: "The rule of thumb leaves 51 kW of surplus on the grid. The optimal plan
   leaves 5.6 kW, and it serves more demand: 86.8% against 85.4%."
4. Point to the table: QAOA matches the exact optimum. Point to the histogram: "Green is the optimal plan. It is
   15 times more likely than a random guess."
5. Move density to 0.6: "Denser grid: more couplings, deeper circuits. This is where hardware noise bites."
6. Backup: if the app or the network fails, play a 60-second screen recording (`pitch/demo_backup.mp4`). Record
   it the night before.

## Rehearsal checklist

* [ ] Two full timed run-throughs out loud. Cut words, not slides, if over 5:00.
* [ ] Rehearse the slide-to-app switch and back until it takes under 3 seconds.
* [ ] Laptop on power, notifications off, browser zoom at 125%, app already running.
* [ ] Phone timer visible on the desk. Glance at it when you reach the results slide. It should read about 3:15.
