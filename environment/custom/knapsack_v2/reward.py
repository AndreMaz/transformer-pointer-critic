from environment.custom.knapsack_v2.misc.utils import bins_eos_checker

def RewardFactory(opts: dict, EOS_NODE):
    rewards = {
        "greedy": GreedyReward,
        # ToDO: Add more rewards
    }

    try:
        rewardType = opts['type']
        R = rewards[f'{rewardType}']
        # print(f'"{rewardType.upper()}" reward selected.')
        return R(opts[f'{rewardType}'], EOS_NODE)
    except KeyError:
        raise NameError(f'Unknown Reward Name! Select one of {list(rewards.keys())}')

class GreedyReward():
    def __init__(self, opts: dict, EOS_NODE):
        super(GreedyReward, self).__init__()
        self.EOS_NODE = EOS_NODE
    
    def compute_reward(self,
                       updated_batch,
                       original_batch,
                       total_num_nodes,
                       bins,
                       items,
                       feasible_mask,
                       node_ids,
                       ):

        batch_size = updated_batch.shape[0]
        num_features = updated_batch.shape[2]

        is_eos = bins_eos_checker(bins, self.EOS_NODE[0], num_features, "float32")

        item_values = items[:, 1]

        reward = item_values * (1 - is_eos)

        return reward