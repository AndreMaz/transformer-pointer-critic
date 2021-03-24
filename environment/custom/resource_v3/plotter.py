# For plotting
# from environment.custom.knapsack.heuristic import solver
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_attentions(attentions,
                    num_resources,
                    num_bins
                ):
    
    fig, axs = plt.subplots(num_resources, 2)

    for index, attention in enumerate(attentions):
        
        # Only show the attention over the resources
        resource_attention = attention['resource_attention'][:, num_bins:]
        axs[index, 0].matshow(np.transpose(resource_attention))
        # axs[index, 0].set_title('Item Attention')

        # Only show the attention over the bins
        bin_attention = attention['bin_attention'][:, :num_bins]
        axs[index, 1].matshow(np.transpose(bin_attention))
        # axs[index, 1].set_title('Backpack Attention')

    for index in range(num_resources):
        # Select the plot by index for the Items
        plt.sca(axs[index, 0])
        # Add the ticks and the labels
        resource_input = attentions[index]["resource_net_input"]

        CPU = int(round(resource_input[0,0,0]  * 100))
        RAM = int(round(resource_input[0,0,1]  * 100))
        MEM = int(round(resource_input[0,0,2]  * 100))

        resource_xlabel = f'C:{CPU} R:{RAM} M:{MEM}'
        plt.xticks([0], [resource_xlabel], fontsize=8)

        resource_states = attentions[index]['current_state'][0, num_bins:]
        resource_ylabel = []
        for itm in resource_states:
            CPU = int(round(itm[0] * 100))
            RAM = int(round(itm[1] * 100))
            MEM = int(round(itm[2] * 100))


            resource_ylabel.append(
                f'C:{CPU} R:{RAM} M:{MEM}'
            )

        plt.yticks(range(len(resource_ylabel)), resource_ylabel, rotation=0, fontsize=8)

        # Select the plot by index for the Backpacks
        plt.sca(axs[index, 1])
        # Add the ticks and the labels
        resource_input = attentions[index]["bin_net_input"]
        CPU = int(round(resource_input[0,0,0] * 100))
        RAM = int(round(resource_input[0,0,1] * 100))
        MEM = int(round(resource_input[0,0,2] * 100))

        bin_xlabel = f'C:{CPU} R:{RAM} M:{MEM}'
        plt.xticks([0], [bin_xlabel], fontsize=8)

        bin_states = attentions[index]['current_state'][0, :num_bins]
        bin_ylabel = []
        for bp in bin_states:
            CPU = int(round(bp[0] * 100)  )
            RAM = int(round(bp[1] * 100)  )
            MEM = int(round(bp[2] * 100)  )

            bin_ylabel.append(
                f'C:{CPU} R:{RAM} M:{MEM}'
            )
        plt.yticks(range(len(bin_ylabel)), bin_ylabel, rotation=0, fontsize=8)
    
    # plt.subplots_adjust(wspace=0.3, hspace = 0.3)
    plt.tight_layout()
    plt.show(block=True)
