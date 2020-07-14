# Common stuff
from agents.common.tester import test
from agents.common.plotter import plotter

# Double Pointer Critic
from agents.double_pointer_critic.agent import DoublePointerCritic as DoublePointerCritic
from agents.double_pointer_critic.trainer import trainer as DoublePointerCriticTrainer

# Double Transfomer Pointer Critic
from agents.transformer_pointer_critic.agent import TransfomerPointerCritic
from agents.transformer_pointer_critic.trainer import trainer as TransfomerPointerCriticTrainer

agents = {
    'dpc': (DoublePointerCritic, DoublePointerCriticTrainer, test, plotter),
    'tpc': (TransfomerPointerCritic, TransfomerPointerCriticTrainer, test, plotter),
}


def agent_factory(name, opts):
    try:
        agent, trainer, tester, plotter = agents[name]
        print(f'"{name.upper()}" agent selected.')
        return agent(name, opts), trainer, tester, plotter
    except KeyError:
        raise NameError(f'Unknown Agent Name! Select one of {list(agents.keys())}')


if __name__ == "__main__":
    pass
