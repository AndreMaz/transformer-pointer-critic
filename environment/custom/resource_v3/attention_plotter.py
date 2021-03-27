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

        CPU = int(round(resource_input[0,0,0]  * 100))
        RAM = int(round(resource_input[0,0,1]  * 100))
        MEM = int(round(resource_input[0,0,2]  * 100))

        resource_label = f'C:{CPU} R:{RAM} M:{MEM}'
        plt.xticks([0], [resource_label], fontsize=8)

        node_states = attentions[index]['current_state'][0, :num_bins]
        nodes_label = []
        for index, node in enumerate(node_states):
            if index == 0:
                CPU = int(round(node[0]))
                RAM = int(round(node[1]))
                MEM = int(round(node[2]))
            else:
                CPU = int(round(node[0] * 100))
                RAM = int(round(node[1] * 100))
                MEM = int(round(node[2] * 100))

            nodes_label.append(
                f'C:{CPU} R:{RAM} M:{MEM}'
            )

        plt.yticks(range(len(nodes_label)), nodes_label, rotation=0, fontsize=8)

    # plt.subplots_adjust(wspace=0.3, hspace = 0.3)
    plt.tight_layout()
    plt.show(block=True)

if  __name__ == "__main__": # pragma: no cover
    current_state = np.array([[
        [-2.  , -2.  , -2.  , -2.  ],
        [ 0.05,  0.  ,  0.04,  0.  ],
        [ 0.05,  0.02,  0.01,  0.  ],
        [ 0.03,  0.07,  0.04,  0.  ],
        [ 0.08,  0.09,  0.08,  0.  ],
        [ 0.05,  0.  ,  0.06,  0.  ],
        [ 0.1 ,  0.2 ,  0.13,  0.  ],
        [ 0.01,  0.09,  0.04,  0.  ],
        [ 0.21,  0.19,  0.19,  0.  ],
        [ 0.28,  0.09,  0.06,  0.  ],
        [ 0.11,  0.03,  0.26,  0.  ],
        [ 0.29,  0.19,  0.24,  0.  ],
        [ 0.11,  0.02,  0.21,  0.  ],
        [ 0.19,  0.18,  0.23,  0.  ],
        [ 0.29,  0.23,  0.15,  0.  ],
        [ 0.18,  0.1 ,  0.25,  0.  ]]], dtype="float32")
    
    decoder_input = np.array([[[0.01, 0.09, 0.04]]], dtype="float32")

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