import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
from .workload_generator import WorkloadGenerator, WorkloadCharacteristics
from .lsm.cost import Cost
from .lsm.types import System, LSMDesign, Policy, Workload

def run_cost_experiments(epsilons: List[float], rhos: List[float], num_trials: int = 5) -> List[Dict]:
    """Run experiments using the actual cost function with different privacy and robustness levels."""
    results = []
    cost_model = Cost(max_levels=10)
    
    # Base workload characteristics
    base_characteristics = WorkloadCharacteristics(
        read_ratio=0.7,
        write_ratio=0.3,
        key_size=16,
        value_size=100,
        operation_count=100000,
        hot_key_ratio=0.2,
        hot_key_count=100
    )
    
    # Base system configuration
    base_system = System(
        entry_size=base_characteristics.key_size + base_characteristics.value_size,
        selectivity=0.1,
        entries_per_page=128,
        num_entries=base_characteristics.operation_count,
        mem_budget=1000,
        phi=1.0
    )
    
    # Base LSM design
    base_design = LSMDesign(
        bits_per_elem=10,
        size_ratio=10,
        policy=Policy.Classic,
        kapacity=()
    )
    
    for epsilon in epsilons:
        for rho in rhos:
            print(f"Running experiments for ε={epsilon}, ρ={rho}")
            
            for trial in range(num_trials):
                # Create workload generator with current epsilon
                workload_generator = WorkloadGenerator(epsilon=epsilon)
                
                # Generate workloads with privacy
                original_workload, private_workload = workload_generator.generate_workload(base_characteristics)
                
                # Apply robustness factor (rho) to the system parameters
                system = System(
                    entry_size=base_system.entry_size,
                    selectivity=base_system.selectivity * (1 + rho),  # Increase selectivity with robustness
                    entries_per_page=base_system.entries_per_page,
                    num_entries=base_system.num_entries,
                    mem_budget=int(base_system.mem_budget * (1 + rho)),  # Increase memory budget with robustness
                    phi=base_system.phi
                )
                
                # Apply robustness factor to the LSM design
                design = LSMDesign(
                    bits_per_elem=base_design.bits_per_elem * (1 + rho),  # More bits per element for robustness
                    size_ratio=base_design.size_ratio,
                    policy=Policy.Classic,
                    kapacity=()
                )
                
                # Convert workloads to Endure format
                private_reads = [op for op in private_workload if op["type"] == "read"]
                private_writes = [op for op in private_workload if op["type"] == "write"]
                
                # Calculate costs for read and write operations
                read_workload = Workload(
                    z0=sum(1 for op in private_reads if not op.get("is_hot", False)) / len(private_reads) if private_reads else 0,
                    z1=sum(1 for op in private_reads if op.get("is_hot", False)) / len(private_reads) if private_reads else 0,
                    q=0.1,
                    w=0.0
                )
                
                write_workload = Workload(
                    z0=0.0,
                    z1=0.0,
                    q=0.0,
                    w=1.0
                )
                
                # Calculate costs
                read_cost = cost_model.calc_cost(design, system, read_workload) if private_reads else 0
                write_cost = cost_model.calc_cost(design, system, write_workload) if private_writes else 0
                total_cost = read_cost + write_cost
                
                result = {
                    "epsilon": epsilon,
                    "rho": rho,
                    "trial": trial,
                    "dp_metrics": {
                        "total_cost": total_cost,
                        "read_cost": read_cost,
                        "write_cost": write_cost
                    },
                    "system_params": {
                        "selectivity": system.selectivity,
                        "mem_budget": system.mem_budget,
                        "bits_per_elem": design.bits_per_elem
                    }
                }
                results.append(result)
    
    return results

def plot_cost_analysis(save_results: bool = True):
    """Plot cost analysis using actual cost function experiments."""
    # Define experiment parameters
    epsilons = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    rhos = [0.1, 0.3, 0.5, 0.7]
    
    # Ensure the results directory exists
    results_path = Path("endure/dp_experiment/results")
    results_path.mkdir(parents=True, exist_ok=True)
    results_file = results_path / "robust_experiment_summary.json"
    
    # Run experiments or load existing results
    if not results_file.exists() or save_results:
        print("Running cost function experiments...")
        results = run_cost_experiments(epsilons, rhos)
        if save_results:
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
    else:
        print("Loading existing results...")
        with open(results_file, "r") as f:
            results = json.load(f)
    
    # Aggregate results by epsilon and rho
    epsilons = sorted(set(entry["epsilon"] for entry in results))
    rhos = sorted(set(entry["rho"] for entry in results))
    avg_costs = np.zeros((len(rhos), len(epsilons)))
    std_costs = np.zeros((len(rhos), len(epsilons)))

    for i, rho in enumerate(rhos):
        for j, epsilon in enumerate(epsilons):
            costs = [
                entry["dp_metrics"]["total_cost"]
                for entry in results
                if entry["epsilon"] == epsilon and entry["rho"] == rho
            ]
            avg_costs[i, j] = np.mean(costs)
            std_costs[i, j] = np.std(costs) if len(costs) > 1 else 0

    # Create main plot
    plt.figure(figsize=(12, 8))
    
    # Plot average costs with error bands
    for i, rho in enumerate(rhos):
        plt.plot(epsilons, avg_costs[i, :], marker='o', label=f'ρ={rho}', linewidth=2)
        plt.fill_between(
            epsilons,
            avg_costs[i, :] - std_costs[i, :],
            avg_costs[i, :] + std_costs[i, :],
            alpha=0.2
        )

    # Add privacy regions
    plt.axvspan(0.1, 1.0, alpha=0.1, color='green', label='High Privacy')
    plt.axvspan(1.0, 3.0, alpha=0.1, color='yellow', label='Medium Privacy')
    plt.axvspan(3.0, 5.0, alpha=0.1, color='red', label='Low Privacy')

    plt.xlabel('ε (Epsilon)', fontsize=12)
    plt.ylabel('Average Total Cost', fontsize=12)
    plt.title('Cost vs. Privacy Parameter (ε) for Different Robustness Levels (ρ)', fontsize=14)
    plt.legend(title='Robustness Level', title_fontsize=12, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations for privacy regions
    plt.annotate('High\nPrivacy', xy=(0.5, plt.ylim()[1]), ha='center', va='bottom')
    plt.annotate('Medium\nPrivacy', xy=(2.0, plt.ylim()[1]), ha='center', va='bottom')
    plt.annotate('Low\nPrivacy', xy=(4.0, plt.ylim()[1]), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('cost_vs_epsilon_by_rho_analysis.png', dpi=300, bbox_inches='tight')
    print("Generated visualization: cost_vs_epsilon_by_rho_analysis.png")

if __name__ == "__main__":
    plot_cost_analysis() 