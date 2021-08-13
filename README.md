# Double Transformer Pointer-Critic

**For the rationale behind the architecture please check [Architecture Rationale](./Arch_Rationale.md)**

## Architecture Details

**Simple Overview**
![simple_arch](./media/paper_arch.png)

### Goal

The goal is to design a load balancing strategy that's able to distribute the incoming requests across the devices. The designed strategy must prioritize the `premium` requests but at the same time satisfy the `free` requests. By providing good quality-of-service for the `free` users it's more probable for them to become `premium` and recommend the system to other users.

### Input Representation

```python
array([
    [ 0., 0., 0., 0., 0.],  -> Node EOS. Rejected items will be "placed" here
    [ 70., 80., 40., 4., 7.] -> Node 1. Remaining CPU: 70 | Remaining RAM: 80 | Remaining Memory: 40 | Tasks without penalty `4`, `5`, `6`, `7`
    [ 50., 40., 20., 1., 4.] -> Node 2. Remaining CPU: 50 | Remaining RAM: 40 | Remaining Memory: 20 | Tasks without penalty `1`, `2`, `3`, `4`
    [ 10., 12., 17., 3., 0.] -> Request 1. Required CPU: 10 | Required RAM: 12 | Required Memory: 17 | Task: 3 | User Type: 0 (`free`)
    [ 18., 32., 16., 4., 1.] -> Request 2. Required CPU: 18 | Required RAM: 32 | Required Memory: 16 | Task: 4 | User Type: 1 (`premium`)
    ],
    dtype=float32, shape=(5, 5))
```

### Training

### Testing

## Useful Links

- [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)
- [Deriving Policy Gradients and Implementing REINFORCE](https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63)
- [Understanding Actor Critic Methods and A2C](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f)
- [Beam Search](https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/)

### Pointer Critic

- [Neural Combinatorial Optimization with Reinforcement Learning](https://arxiv.org/pdf/1611.09940.pdf)
- [Presentation Video - Neural Combinatorial Optimization with Reinforcement Learning](https://www.youtube.com/watch?v=mxCVgVrUw50)
- [Reviews - Neural Combinatorial Optimization with Reinforcement Learning](https://openreview.net/forum?id=rJY3vK9eg)
- [Reinforcement Learning for Solving the Vehicle Routing Problem](https://arxiv.org/pdf/1802.04240.pdf)
- [Order Matters: Sequence to sequence for sets](https://arxiv.org/pdf/1511.06391.pdf)
- [Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475)

### Unit Test and Coverage

```bash
python environment/custom/resource/tests/runner.py
```

or to generate an HTML-based coverage file

```bash
coverage run tests/runner.py
coverage html --omit=*/venv/*,*/usr/*,*/lib/*,*/tests/* -i
```

or combo

```
coverage run tests/runner.py && coverage html --omit=*/venv/*,*/usr/*,*/lib/*,*/tests/* -i
```

## Potential Improvements and Interesting ToDos

### Implement Self-Critic

Instead of using a dedicated network (the `Critic`) to estimate the state-values, which are used as a baseline, use [greedy rollout baseline](https://arxiv.org/abs/1612.00563). Greedy rollout baseline in [Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475) show promising results.

## How to do it

The easiest (not the cleanest) way to implement it is to create a `agents/baseline_trainer.py` file with a 2 instances (`env` and `env_baseline`) of environment and agents (`agent` and `agent_baseline`).

Then:

- When we sample a state from `env` we would copy it state into `env_baseline`.
- Delete the `critic` model from `agent` and `agent_baseline` as it's no longer necessary.
- Copy the network weighs for `agent` actor into `agent_baseline` actor.
- Set `agent_baseline.stochastic_action_selection` to `False`. This way the agent will select the action in a greedy way.
- The `agent` would gather rewards from `env` and `agent_baseline` would do the same with `env_baseline`.

### Implement Vehicle Routing Problem environment

It would be interesting to see how the network performs in VRP

## How to do it

- Look at the `Knapsack` and `Resource` environments in `environments/custom` and adapt them to the VRP
- Add the VRP env to `environments/env_factory.py`
- Add the `JSON` config file into the `configs` folder.
