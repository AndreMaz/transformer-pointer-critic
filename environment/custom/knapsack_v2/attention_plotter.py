# For plotting
# from environment.custom.knapsack.heuristic import solver
import matplotlib.pyplot as plt
import os
import numpy as np

def attention_plotter(attentions,
                    num_resources,
                    num_bins
                ):
    
    fig, axs = plt.subplots(1, num_resources)
    # fig.suptitle('Attentions')

    #for ax in axs.flat:
    #    ax.set(xlabel='resource', ylabel='nodes')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    #for ax in axs.flat:
    #    ax.label_outer()

    for index, attention in enumerate(attentions):
        # Only show the attention over the bins
        bin_attention = attention['attention_probs'][:, :num_bins]
        axs[index].matshow(np.transpose(bin_attention))
        # axs.matshow(np.transpose(bin_attention))
        # axs[index].set_title('Backpack Attention')

    for index in range(num_resources):
        # Select the plot by index for the Items
        plt.sca(axs[index])
        # Add the ticks and the labels
        resource_input = attentions[index]["decoder_input"]

        weight = int(round(resource_input[0,0,0]  * 100))
        value = int(round(resource_input[0,0,1]  * 100))

        resource_label = f'W:{weight} V:{value}'
        plt.xticks([0], [resource_label], fontsize=8)

        node_states = attentions[index]['current_state'][0, :num_bins]
        nodes_label = []
        for index, bin in enumerate(node_states):
            if index == 0:
                capacity = int(round(bin[0]))
                load = int(round(bin[1]))
            else:
                capacity = int(round(bin[0] * 100))
                load = int(round(bin[1] * 100))

            nodes_label.append(
                f'C:{capacity} L:{load}'
            )

        plt.yticks(range(len(nodes_label)), nodes_label, rotation=0, fontsize=8)

    # plt.subplots_adjust(wspace=0.3, hspace = 0.3)
    plt.tight_layout()
    plt.show(block=True)

if  __name__ == "__main__": # pragma: no cover
    current_state = np.array([[
        [-2.  , -2.  ],
        [ 0.05,  0.  ],
        [ 0.05,  0.02],
        [ 0.03,  0.07],
        [ 0.08,  0.09],
        [ 0.05,  0.  ],
        [ 0.1 ,  0.2 ],
        [ 0.01,  0.09],
        [ 0.21,  0.19],
        [ 0.28,  0.09],
        [ 0.11,  0.03],
        [ 0.29,  0.19],
        [ 0.11,  0.02],
        [ 0.19,  0.18],
        [ 0.29,  0.23],
        [ 0.18,  0.1 ]]], dtype="float32")
    
    decoder_input = np.array([[[0.01, 0.09]]], dtype="float32")

    attention_probs = np.array(
        [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
      dtype="float32")

    attentions = []

    attentions.append({
        "current_state": current_state,
        "decoder_input": decoder_input,
        "attention_probs": attention_probs
    })

    num_nodes = 6
    num_resources = 10

    attention_plotter(attentions, num_resources, num_nodes)