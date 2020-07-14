# Double Pointer Critic
from agents.double_pointer_critic.agent import DoublePointerCritic as DoublePointerCritic
from agents.double_pointer_critic.trainer import trainer as DoublePointerCriticTrainer
from agents.double_pointer_critic.tester import test as DoublePointerCriticTester
from agents.double_pointer_critic.plotter import plotter as DoublePointerCriticPlotter

# Double Transfomer Pointer Critic
from agents.transformer_pointer_critic.agent import TransfomerPointerCritic
from agents.transformer_pointer_critic.trainer import trainer as TransfomerPointerCriticTrainer
from agents.transformer_pointer_critic.tester import test as TransfomerPointerCriticTester
from agents.transformer_pointer_critic.plotter import plotter as TransfomerPointerCriticPlotter

agents = {
    'dpc': (DoublePointerCritic, DoublePointerCriticTrainer, DoublePointerCriticTester, DoublePointerCriticPlotter),
    'tpc': (TransfomerPointerCritic, TransfomerPointerCriticTrainer, TransfomerPointerCriticTester, TransfomerPointerCriticPlotter),
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
