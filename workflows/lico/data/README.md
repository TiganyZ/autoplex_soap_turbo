# Seed data

`make_seed.py` writes `seed.xyz` (gitignored): crystalline Li₂CO₃ (zabuyelite,
C2/c, a 8.359, b 4.973, c 6.198 Å, β 114.83°, Z = 4), five isotropic strains of
±4 %, four random shears up to 3 %, six rattles of 0.05–0.15 Å, and one rattled
1×2×1 supercell — 16 cells of 24–48 atoms. No DFT labels: the flow computes them.

`--seed N` changes the shears and rattles; the strained cells are the same for
every seed.
