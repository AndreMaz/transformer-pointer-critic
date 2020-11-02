
def RewardFactory(opts: dict, penalizer):
    rewards = {
        'greedy': GreedyReward,
    }

    try:
        rewardType = opts['type']
        R = rewards[f'{rewardType}']
        return R(opts[f'{rewardType}'], penalizer)
    except KeyError:
        raise NameError(f'Unknown Reward Name! Select one of {list(rewards.keys())}')


class GreedyReward():
    def __init__(self,
                 opts: dict,
                 penalizer
                 ):
        super(GreedyReward, self).__init__()

        self.reward_per_level = opts['reward_per_level']
        self.misplace_reward_penalty = opts['misplace_reward_penalty']
        self.penalizer = penalizer

    def compute_reward(self,
                       batch,
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
        request_type = int(resource[4])
        
        reward = 0
        if self.penalizer.to_penalize(bin_lower_type, bin_upper_type, resource_type):
            reward = self.reward_per_level[request_type] - self.misplace_reward_penalty
        else:
            reward = 10 * self.reward_per_level[request_type]
            
        return reward
