import json
import matplotlib.pyplot as plt
import numpy as np

def load_json_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def prepare_data(data):
    epsilons = sorted([float(k) for k in data.keys()])
    metrics = ['throughput', 'latency', 'space_amplification']
    
    original_data = {m: {e: [] for e in epsilons} for m in metrics}
    private_data = {m: {e: [] for e in epsilons} for m in metrics}
    
    for epsilon, runs in data.items():
        e = float(epsilon)
        for run in runs:
            for metric in metrics:
                original_data[metric][e].append(run['original_config'][metric])
                private_data[metric][e].append(run['private_config'][metric])
    
    return epsilons, original_data, private_data

def create_box_plot(epsilons, original_data, private_data, metric, output_file):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    width = 0.35
    x = np.arange(len(epsilons))
    
    original_values = [original_data[metric][e] for e in epsilons]
    private_values = [private_data[metric][e] for e in epsilons]
    
    bp1 = ax.boxplot(original_values, positions=x - width/2, widths=width/2, 
                     patch_artist=True, boxprops=dict(facecolor='lightblue'))
    bp2 = ax.boxplot(private_values, positions=x + width/2, widths=width/2,
                     patch_artist=True, boxprops=dict(facecolor='lightcoral'))
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in epsilons])
    ax.set_xlabel('Epsilon')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{metric.capitalize()} Comparison: Original vs Private Config')
    
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Original', 'Private'])
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main(file_path):
    data = load_json_data(file_path)
    epsilons, original_data, private_data = prepare_data(data)
    
    metrics = ['throughput', 'latency', 'space_amplification']
    for metric in metrics:
        output_file = f'{metric}_comparison.png'
        create_box_plot(epsilons, original_data, private_data, metric, output_file)

if __name__ == '__main__':
    main('json1.json')