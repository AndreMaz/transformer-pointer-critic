

def compute_reward(batch,
                   total_num_nodes,
                   bin,
                   resource,
                   ):
    bins = batch[:total_num_nodes]
    resources = batch[total_num_nodes:]

    bin_remaning_CPU = bin[0]
    bin_remaning_RAM = bin[1]
    bin_remaning_MEM = bin[2]
    bin_lower_type = bin[3]
    bin_upper_type = bin[4]

    resource_CPU = resource[0]
    resource_RAM = resource[1]
    resource_MEM = resource[2]
    resource_type = resource[3]
    request_type = resource[4]

    return 1
