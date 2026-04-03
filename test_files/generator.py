#!/usr/bin/env python3

import random
import os
import sys

# Each test is a dict with:
#   name: output filename (without wave suffix)
#   num_particles: number of particles per wave
#   num_waves: number of wave files to generate
#   pos_range: (min, max) inclusive range for positions
#   energy_range: (min, max) inclusive range for energies
#   seed: base random seed (each wave adds its index)

TESTS = [
    {
        "name": "test_personal_01_a60k_p20k",
        "num_particles": 20000,
        "num_waves": 6,
        "pos_range": (1, 59999),
        "energy_range": (50, 149),
        "seed": 4201,
    },
    {
        "name": "test_personal_02_a120k_p20k",
        "num_particles": 20000,
        "num_waves": 6,
        "pos_range": (1, 119999),
        "energy_range": (50, 149),
        "seed": 4202,
    },
    {
        "name": "test_personal_03_a2M_p5k",
        "num_particles": 5000,
        "num_waves": 4,
        "pos_range": (1, 1999999),
        "energy_range": (500000, 1000000),
        "seed": 4203,
    },
    {
        "name": "test_personal_04_a4M_p5k",
        "num_particles": 5000,
        "num_waves": 4,
        "pos_range": (1, 3999999),
        "energy_range": (500000, 1000000),
        "seed": 4204,
    },
]

def generate_wave(test_cfg, wave_index, output_dir):

    rng = random.Random(test_cfg["seed"] + wave_index)

    n = test_cfg["num_particles"]
    pmin, pmax = test_cfg["pos_range"]
    emin, emax = test_cfg["energy_range"]

    particles = []
    for _ in range(n):
        pos = rng.randint(pmin, pmax)
        energy = rng.randint(emin, emax)
        particles.append((pos, energy))

    filename = f"{test_cfg['name']}_w{wave_index}"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        f.write(f"{n}\n")
        for pos, energy in particles:
            f.write(f"{pos} {energy}\n")

    return filepath


def main():
    output_dir = os.environ.get("OUTPUT_DIR", ".")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for test_cfg in TESTS:
        print(f"\n=== {test_cfg['name']} ===")
        print(f"    particles: {test_cfg['num_particles']}")
        print(f"    waves:     {test_cfg['num_waves']}")
        print(f"    positions: [{test_cfg['pos_range'][0]}, {test_cfg['pos_range'][1]}]")
        print(f"    energies:  [{test_cfg['energy_range'][0]}, {test_cfg['energy_range'][1]}]")

        for w in range(1, test_cfg["num_waves"] + 1):
            path = generate_wave(test_cfg, w, output_dir)
            print(f"    -> {path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
