import os
import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(problem_instances, metrics):
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    names = ['Our Approach', 'A*']
    approaches = ['A* w/ EARL', 'A*']
    num_approaches = len(approaches)    
    
    for idx, problem_instance in enumerate(problem_instances):
        earl = metrics[idx][approaches[0]]
        a_star = metrics[idx][approaches[1]]   
        x_values = np.arange(num_approaches)

        plt.bar(x_values[0], earl, width=0.4, label='Our Approach')
        plt.bar(x_values[1], a_star, width=0.4, label='A*')
        plt.xlabel(problem_instance.capitalize())
        plt.xticks(x_values, labels=names)
        plt.ylim(0, 10)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{results_dir}/{problem_instance}.png')
        plt.clf()
        
        print(f'{problem_instance}: {metrics[idx]}')
        
        