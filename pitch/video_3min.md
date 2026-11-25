# 3-minute submission video: script (Team Qilowatt, solo)

Judges pick the Top 10 from the repository and this video, before the live final. Target about 430 spoken words
(roughly 145 words per minute). Record the screen sections at 1080p. Start the app with
`streamlit run app/streamlit_app.py`, use its defaults: n = 8, density 0.3, seed 5, p = 2.

Numbers come from `results/summary.md`. Only the bracketed statistic is still to fill. Do not round in your favour.

| Time | On screen | Voice-over |
|---|---|---|
| 0:00–0:20 | Your face on camera for one sentence, then a night photo of a village or island (credit the source), then the app's grid map with every node hollow | "Tonight, at the end of a rural feeder or on a small island, someone running a microgrid decides which batteries, solar inverters and diesel generators to switch on for the next hour. Get it wrong one way and families sit in the dark. Get it wrong the other way and clean energy is thrown away, or diesel is burnt for nothing." |
| 0:20–0:50 | Title card: "Team Qilowatt · Climate, Energy & Agriculture". One statistic slide with its source | "I'm **[your name]**, and I'm Team Qilowatt. **[One verified statistic about the community, with source.]** The users are the cooperatives and utilities that run small grids, and the homes, clinics and water pumps on them. So I measure what they care about: demand met, energy wasted, diesel used and cost." |
| 0:50–1:15 | `docs/architecture.png` | "Every node becomes one qubit: switch its resource on or off. Generation cost plus a local balance penalty, with a 15% reserve, becomes a QUBO, then an Ising Hamiltonian, and I solve it with QAOA in Qiskit. Sources only interact if they can reach the same village. That means sparse grids give sparse circuits." |
| 1:15–1:40 | Screen recording: move the seed slider, show the map light up, the comparison table and the histogram | "I train the circuit once on a simulator and warm-start each new layer. Then I reuse the same parameters on a noise model of IBM's Heron processor and on real hardware, so hardware time goes only to sampling. Here it picks which resources to switch on. Green bars are the optimal plan." |
| 1:40–2:30 | `results/figures/baselines.png`, then a slide with the shots-for-95% table from the README | "I tested 36 grids of 6 to 12 nodes against brute force, simulated annealing and the operator's rule of thumb: cheapest first. That rule finds the best plan in only 8 of 36 grids, and leaves 59% more surplus power on the grid. QAOA found the optimal plan in all 36, with and without Heron noise. The fair measure is shots: at 12 qubits, being 95% sure of hitting the optimal plan takes about 12,000 random guesses, 425 QAOA shots noise-free, and 1,400 under Heron noise. And to be clear: at this size, simulated annealing is faster. I'm not claiming a speed-up. I'm showing the full pipeline running on hardware, and where noise and grid density limit it today." |
| 2:30–3:00 | Next-steps slide, then the repo URL and QR code. Your face again for the last line | "Next comes multi-hour planning with battery charge, and grids of 50 to 100 nodes, where brute force stops working. Everything is open source and runs on a laptop today. I'd love a partner microgrid with real load data. Thank you." |

## Recording checklist

* [ ] Face on camera for the first and last sentence. Screen and slides for the rest.
* [ ] Subtitles burned in. Judges often watch muted.
* [ ] Final length between 2:50 and 3:00. Over 3:00 risks being cut off.
* [ ] Record the voice-over in short takes, one table row per take, then join them. This is faster than one perfect take.
* [ ] Upload as unlisted on YouTube or Drive with "anyone with the link" access, then test the link in a private window.
* [ ] Put the link at the top of the README and in the submission form.
