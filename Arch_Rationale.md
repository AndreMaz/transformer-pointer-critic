
# Contents
- [Traveling Salesman Problem](#Traveling-Salesman-Problem)
- [Knapsack Problem](#Knapsack-Problem)
- [Multiple Knapsack Problem](#Multiple-Knapsack-Problem)
- [Resource Placement at Edge Devices](#Resource-Placement-At-Edge-Devices)

# Traveling Salesman Problem

**Problem statement**: Given a set of nodes, each located at {`x`, `y`}, the goal is to visit them in a way that the total traveling distance is minimized.

**Claim**: Given a set of nodes, it is possible to sort them in a ___specific___ way where the order by which the nodes were sorted represents the (near) optimal visitation sequence.

**Goal of Pointer-Network** Given a set of nodes, the goal of the Pointer-Network is to generate a sequence by which the nodes will be visited.

**Encoder Input** Represents the coordinates of the nodes

```bash
array([
    [ 0.,  0.],  -> Node 0. Location (0, 0). Also the starting point
    [ 1.,  1.],  -> Node 1. Location (1, 1) 
    [ 8.,  2.],  -> Node 2. Location (8, 2) 
    [ 5.,  0.],  -> Node 3. Location (5, 0) 
    ],
```


**Decoder Input** Represents the current previous location of the salesman. At the beginning it will be at the starting point.

```bash
array([
    [ 0.,  0.],  -> Salesman location at decoding step `0`. Location (0, 0) 
    ],
```

Assuming that at first decoding step the Pointer-Network "pointed" at Node 1, located at (1,1), the decoder's input for the next decoding step would be updated to:

```bash
array([
    [ 1.,  1.],  -> Salesman location at decoding step `1`. It is at the location of the Node 1.
    ],
```

After 4 decoding steps the Pointer-Network would have generated the following sequence: 

```
Node 1 -> Node 3 -> Node 2 -> Node 0
```

This sequence represents the solution to this particular problem. However, if the input of the problem changes the visitation sequence will also change.

## Goal of the Embedding Layers and the Attention
### Encoder's Embedding Layers
In this case in particular the embedding layers will represent, in a high dimensional space, the distance map between the nodes. Two nodes that are close to each other will have similar representations. Nodes that are distant from each other will have very different representations.

### Pointer-Network Attention
Given the encoder's and the decoder's input the attention will try to focus (by giving higher probability) on specific nodes that should be considered during the current decoding step.

# Knapsack Problem
**Problem statement**: Given a set of items, each with a weight `x` and value `y`, and a backpack with a capacity `c` the goal is to take the items in a way that the profit is maximized.

**Claim**: Given a set of items, it is possible to sort them in a ___specific___ way where the order by which the items were sorted represents the (near) optimal picking sequence.

**Goal of Pointer-Network** Given a set of items and a backpack, the goal of the Pointer-Network is to generate a sequence by which the items will be selected and placed in the backpack.

**Encoder Input** Represents the coordinates of the nodes

```bash
array([
    [ EOS, EOS ]   -> Item 0. Pointer-Net will point here if the items no longer fit.
    [ 5.,  5/7.],  -> Item 1. Value `5`. Occupies 5 units of space in a backpack with 7 units of capacity.
    [ 6.,  2/7.],  -> Item 2. Value `6`. Occupies 2 units of space in a backpack with 7 units of capacity.
    [ 7.,  1/7.],  -> Item 3. Value `7`. Occupies 1 units of space in a backpack with 7 units of capacity.
    [ 2.,  3/7.],  -> Item 4. Value `2`. Occupies 3 units of space in a backpack with 7 units of capacity.
    ],
```


**Decoder Input** Represents the previously select item. At the beginning it will be at the [SOS, SOS].

```bash
array([
    [ 0.,  0.],  -> Start the decoding process.
    ],
```

> Note: For more info see [A Pointer Network Based Deep Learning Algorithm
for 0-1 Knapsack Problem](https://ieeexplore.ieee.org/document/8377505)

Assuming that at first decoding step the Pointer-Network "pointed" at Item 2, represented as [7., 1/7],  the decoder's input for the next decoding step would be updated to:

```bash
array([
    [ 7.,  1/7.],  -> Item decoded in the previous decoding step.
    ],
```

After 4 decoding steps the Pointer-Network would have generated the following sequence: 

```
Item 3 -> Item 2 -> Item 4 -> Item 0
```

This sequence represents the solution by which the items should be selected.

## Goal of the Embedding Layers and the Attention
### Encoder's Embedding Layers
In this case in particular the embedding layers will represent, in a high dimensional space, weight to value ration between an item and the backpack.

### Pointer-Network Attention
Given the encoder's and the decoder's input the attention will try to focus (by giving higher probability) on specific items that should selected and placed into the backpack during the current decoding step. In other words, given the remaining items and knowing the last selected item the attention will point to the next item that should be selected.

### Problem with this approach
How to handle the multiple knapsack problem? 

# Multiple Knapsack Problem
**Problem statement**: Given a set of items, each with a weight `x` and value `y`, and a set of backpacks, each with a capacity `c`, the goal is to take the items in a way that the profit is maximized.

**Problems with the previous approach**
- [Previous approach](#Knapsack-Problem) is not able to work with multiple backpacks.

- One possibility to solve the problem is to select a specific backpack from a set and then try to insert all the items. Then, select another backpack and try to insert the remaining items. However, what's the order by which the backpacks should be selected? Sort them descending order by their capacities? Will this approach work well every time? No. For specific set of items this approach will generate bad results.

- What if we invert the problem? Pick a single item and try to place it across multiple backpacks. Then, take another item and repeat the process. In this case, what would be the sequence by which we would select the items? Start by the items with the highest cost? Will this approach work well every time? No. 

- Given the fact that we cannot remove the item from the backpack. The item selection and its placement, into a specific backpack, must be done in a way that "we know" what items are remaining and the state of the backpacks.

- While the items are independent from each other their placement is not. Placing an item at a specific backpack **can and will** affect the way by which the other items are be placed.

**Claim**: Given a set of items and the backpacks, it is possible to sort them in a ___specific___ way after which the fit-first (or any other) approach generates the optimal picking and placing sequence.

**Encoder Input** Represents the state of the backpacks and the items that can be picked.

```bash
array([
    [ 0.,  0.],  -> Backpack EOS. Not selected items will be "placed" here
    [ 7.,  0.],  -> Backpack 1. Capacity: 7     | Current Load: 0
    [ 8.,  0.],  -> Backpack 2. Capacity: 8     | Current Load: 0
    [13.,  0.],  -> Backpack 3. Capacity: 13    | Current Load: 0
    [ 8.,  0.],  -> Backpack 4. Capacity: 8     | Current Load: 0
    [ 6.,  0.],  -> Backpack 5. Capacity: 6     | Current Load: 0
    [ 5., 38.],  -> Item 1. Weight: 5   | Value : 38
    [11., 42.],  -> Item 2. Weight: 11  | Value : 42
    [ 9., 46.],  -> Item 3. Weight: 9   | Value : 46
    [17., 23.],  -> Item 4. Weight: 17  | Value : 23
    [20.,  8.]   -> Item 5. Weight: 20  | Value : 8
    ],
    dtype=float32, shape=(11, 2))
```


**Item Selecting Decoder Input** Represents the previously select item. At the beginning it will be at the [SOS, SOS].

```bash
array([
    [ 0.,  0.],  -> Start the decoding process.
    ],
```


**Backpack Selecting Decoder Input** Represents the item that was selected by previous network.

```bash
array([
    [ 5.,  38.],  -> Start the decoding process.
    ],
```

After 5 decoding steps (we have 5 items to place), the Pointer-Network would have generated the following sequence: 

```
(Item 3, Backpack 1) -> (Item 1, Backpack 2) -> (Item 4, Backpack 5) -> (Item 2, Backpack 1) -> (Item 4, Backpack 3)
```

This sequence represents the solution by which the items should be selected and the location where they should be placed.

## Goal of the Embedding Layers and the Attention
### Encoder's Embedding Layers
In this case in particular the embedding layers will simply represent, in a high dimensional space, the items and the backpacks. Similar items will have similar representations.

### Transformer's Encoder Attention
The self-attention mechanism allows the inputs to interact with each other (“self”) and find out who they should pay more attention to (“attention”). The outputs are aggregates of these interactions and attention scores. In the case of the multiple knapsack problem, the attention will "learn" that specific items and specific backpacks tend to generate higher rewards. Lower probabilities (between the item and the backpack) in the attention will mean that for a specific item the network should not consider trying to insert it into the specific backpack because it will generate bad rewards.

In other words, the encoder's attention is looking for possible (good) contender backpack for each item. Moreover, the contender selection also takes into the account the existence of other items because they are also fed into the encoder.

### Pointer-Network Attention
Given the encoder's and the decoder's input the attention will try to focus (by giving higher probability) on specific items that should selected. In other words, given the remaining items, the state of the backpacks and knowing the last selected item the attention will point to the next item that should be selected.


# Resource Placement at Edge Devices
**Problem statement**: At each time `t` a randomly sized batch of user's requests arrive, each has its own profile that contains information about the amount of resources (e.g., [`10` units of CPU, `2` units of RAM, `5` units for Memory]) that it needs in order to be processed properly. The incoming requests must be placed at a set of available nodes, each having its own processing capabilities (e.g., [`100` units of CPU, `20` units of RAM, `50` units for Memory]). In real world, these nodes usually are located behind a reverse proxy such as NGNIX, Traefik or Moleculer API Gateway. All of them provide load balancing capabilities. NGNIX [offers](http://nginx.org/en/docs/http/load_balancing.html) round-robin, least-connected, ip-hash; Traefik, at this moment, only [supports](https://docs.traefik.io/routing/services/#load-balancing) round-robin method; Moleculer API Gateway [offers](https://moleculer.services/docs/0.14/balancing.html#Built-in-strategies) round-robin, random, CPU usage-based and sharding. These load balancing strategies don't provide optimal solution, it's too expensive too look for it in real-time, they simply follow the selected load balancing strategy. These strategies are fast but the results that they provide can be suboptimal.

**Goal**: The goal is to design another load balancing strategy that's able to distribute the incoming requests in a fair way, i.e., in a way that the incoming requests have similar working conditions.

**Purpose of the Neural-based load balancing strategy**: A Neural-based load balancing strategy can adapt the distribution policy (heuristic) according to the incoming user's requests and the state of the nodes and, thus, offer a better way of placing the requests.