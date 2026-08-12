"""Declarative, reproducible job matrix; importing this module never launches GPUs."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product

BASELINES = ("B0-Base", "B1-Clean-KD", "B2-ADS-KD", "B3-DOGe-KD", "B4-Rewrite-KD")
OURS = ("Ours-Clean", "Ours-ADS", "Ours-DOGe", "Ours-Rewrite")
ABLATIONS = ("No-NN", "NN1-only", "NN2-only", "NN1+NN2")
DEFENSE = {"B0-Base":"clean", "B1-Clean-KD":"clean", "B2-ADS-KD":"ads", "B3-DOGe-KD":"doge", "B4-Rewrite-KD":"rewrite", "Ours-Clean":"clean", "Ours-ADS":"ads", "Ours-DOGe":"doge", "Ours-Rewrite":"rewrite", "No-NN":"clean", "NN1-only":"clean", "NN2-only":"clean", "NN1+NN2":"clean"}

@dataclass(frozen=True)
class Job:
    dataset: str; seed: int; method: str; suite: str

def jobs(datasets: list[str], seeds: list[int], methods: tuple[str, ...], suite: str) -> list[Job]:
    return [Job(dataset, seed, method, suite) for dataset, seed, method in product(datasets, seeds, methods)]

def full_jobs(datasets: list[str], seeds: list[int]) -> list[Job]: return jobs(datasets, seeds, BASELINES + OURS, "full")
def ablation_jobs(datasets: list[str], seeds: list[int]) -> list[Job]: return jobs(datasets, seeds, ABLATIONS, "ablation")
