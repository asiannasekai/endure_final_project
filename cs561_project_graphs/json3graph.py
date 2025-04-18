import json
import matplotlib.pyplot as plt

def load_json_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def prepare_data(data):
    parameters = ['read_ratio', 'hot_key_ratio', 'operation_count']
    param_data = {p: {'values': [], 'impacts': []} for p in parameters}
    
    for param in parameters:
        for entry in data[param]:
            param_data[param]['values'].append(entry['value'])
            param_data[param]['impacts'].append(entry['performance_impact'])
    
    return param_data

def create_line_plot(param_data, param, output_file):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    values = param_data[param]['values']
    impacts = param_data[param]['impacts']
    
    ax.plot(values, impacts, marker='o', linestyle='-', color='blue')
    
    ax.set_xlabel(param.replace('_', ' ').title())
    ax.set_ylabel('Performance Impact')
    ax.set_title(f'Performance Impact vs. {param.replace("_", " ").title()}')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main(file_path):
    data = load_json_data(file_path)
    param_data = prepare_data(data)
    
    parameters = ['read_ratio', 'hot_key_ratio', 'operation_count']
    for param in parameters:
        output_file = f'{param}_performance_impact.png'
        create_line_plot(param_data, param, output_file)

if __name__ == '__main__':
    main('json3.json')