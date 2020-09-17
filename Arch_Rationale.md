
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

### Results

Environment configuration for testing and training:
- Set size of items: `100`
- Item value: random in [`1`, `100`] range
- Item weight: random in [`1`, `20`] range
- Set size of backpacks: `20`
- Backpack capacities: random in [`1`, `20`] range

Training configs:
- Batch size: `32`
- Number of epochs: `5000`
- Total number of problem instances used during training: `32 * 5000 = 160000` 
- Item sample size: `20`
- Backpack sample size: `5`

> **Note:** For building a single problem instance the items and backpacks are randomly sampled from their respective sets

Testing configs:
- Batch size: `32`
- Item sample size: `50`
- Backpack sample size: `10 + 1`. `+ 1` is the empty backpack where items that weren't selected are placed.

```bash
Net 1303.0      | Heuristic 1215.0      | % from Heuristic -7.24
Net 1259.0      | Heuristic 1132.0      | % from Heuristic -11.22
Net 1321.0      | Heuristic 1236.0      | % from Heuristic -6.88
Net 1244.0      | Heuristic 1153.0      | % from Heuristic -7.89
Net 1122.0      | Heuristic 994.0       | % from Heuristic -12.88
Net 1065.0      | Heuristic 949.0       | % from Heuristic -12.22
Net 1195.0      | Heuristic 1069.0      | % from Heuristic -11.79
Net 1333.0      | Heuristic 1222.0      | % from Heuristic -9.08
Net 1259.0      | Heuristic 1136.0      | % from Heuristic -10.83
Net 1234.0      | Heuristic 1083.0      | % from Heuristic -13.94
Net 1204.0      | Heuristic 1079.0      | % from Heuristic -11.58
Net 1243.0      | Heuristic 1104.0      | % from Heuristic -12.59
Net 1423.0      | Heuristic 1312.0      | % from Heuristic -8.46
Net 1341.0      | Heuristic 1242.0      | % from Heuristic -7.97
Net 1373.0      | Heuristic 1213.0      | % from Heuristic -13.19
Net 1035.0      | Heuristic 958.0       | % from Heuristic -8.04
Net 1385.0      | Heuristic 1227.0      | % from Heuristic -12.88
Net 1166.0      | Heuristic 1022.0      | % from Heuristic -14.09
Net 1472.0      | Heuristic 1291.0      | % from Heuristic -14.02
Net 1486.0      | Heuristic 1353.0      | % from Heuristic -9.83
Net 1359.0      | Heuristic 1269.0      | % from Heuristic -7.09
Net 1157.0      | Heuristic 1080.0      | % from Heuristic -7.13
Net 1238.0      | Heuristic 1142.0      | % from Heuristic -8.41
Net 1203.0      | Heuristic 1124.0      | % from Heuristic -7.03
Net 1265.0      | Heuristic 1170.0      | % from Heuristic -8.12
Net 1423.0      | Heuristic 1343.0      | % from Heuristic -5.96
Net 1340.0      | Heuristic 1232.0      | % from Heuristic -8.77
Net 1253.0      | Heuristic 1127.0      | % from Heuristic -11.18
Net 1097.0      | Heuristic 955.0       | % from Heuristic -14.87
Net 1364.0      | Heuristic 1226.0      | % from Heuristic -11.26
Net 1339.0      | Heuristic 1231.0      | % from Heuristic -8.77
Net 1289.0      | Heuristic 1213.0      | % from Heuristic -6.27
```

```bash
Opt 369.0       | Net 352.0 | % from Opt 4.61 || Heuristic 352.0     | % from Opt 4.61
Opt 654.0       | Net 593.0 | % from Opt 9.33 || Heuristic 578.0     | % from Opt 11.62
Opt 443.0       | Net 443.0 | % from Opt 0.00 || Heuristic 374.0     | % from Opt 15.58
Opt 666.0       | Net 637.0 | % from Opt 4.35 || Heuristic 637.0     | % from Opt 4.35
Opt 624.0       | Net 612.0 | % from Opt 1.92 || Heuristic 567.0     | % from Opt 9.13
Opt 606.0       | Net 602.0 | % from Opt 0.66 || Heuristic 565.0     | % from Opt 6.77
Opt 514.0       | Net 497.0 | % from Opt 3.31 || Heuristic 479.0     | % from Opt 6.81
Opt 559.0       | Net 505.0 | % from Opt 9.66 || Heuristic 520.0     | % from Opt 6.98
Opt 683.0       | Net 668.0 | % from Opt 2.20 || Heuristic 636.0     | % from Opt 6.88
Opt 450.0       | Net 415.0 | % from Opt 7.78 || Heuristic 415.0     | % from Opt 7.78
Opt 484.0       | Net 484.0 | % from Opt 0.00 || Heuristic 425.0     | % from Opt 12.19
Opt 494.0       | Net 477.0 | % from Opt 3.44 || Heuristic 457.0     | % from Opt 7.49
Opt 664.0       | Net 648.0 | % from Opt 2.41 || Heuristic 583.0     | % from Opt 12.20
Opt 532.0       | Net 515.0 | % from Opt 3.20 || Heuristic 496.0     | % from Opt 6.77
Opt 657.0       | Net 614.0 | % from Opt 6.54 || Heuristic 587.0     | % from Opt 10.65
Opt 661.0       | Net 661.0 | % from Opt 0.00 || Heuristic 606.0     | % from Opt 8.32
Opt 500.0       | Net 500.0 | % from Opt 0.00 || Heuristic 468.0     | % from Opt 6.40
Opt 613.0       | Net 601.0 | % from Opt 1.96 || Heuristic 601.0     | % from Opt 1.96
Opt 417.0       | Net 415.0 | % from Opt 0.48 || Heuristic 378.0     | % from Opt 9.35
Opt 639.0       | Net 574.0 | % from Opt 10.17 || Heuristic 574.0     | % from Opt 10.17
Opt 583.0       | Net 575.0 | % from Opt 1.37 || Heuristic 575.0     | % from Opt 1.37
Opt 592.0       | Net 584.0 | % from Opt 1.35 || Heuristic 486.0     | % from Opt 17.91
Opt 560.0       | Net 552.0 | % from Opt 1.43 || Heuristic 502.0     | % from Opt 10.36
Opt 650.0       | Net 594.0 | % from Opt 8.62 || Heuristic 559.0     | % from Opt 14.00
Opt 599.0       | Net 555.0 | % from Opt 7.35 || Heuristic 507.0     | % from Opt 15.36
Opt 673.0       | Net 630.0 | % from Opt 6.39 || Heuristic 611.0     | % from Opt 9.21
Opt 623.0       | Net 613.0 | % from Opt 1.61 || Heuristic 583.0     | % from Opt 6.42
Opt 841.0       | Net 793.0 | % from Opt 5.71 || Heuristic 747.0     | % from Opt 11.18
Opt 464.0       | Net 446.0 | % from Opt 3.88 || Heuristic 464.0     | % from Opt 0.00
Opt 525.0       | Net 500.0 | % from Opt 4.76 || Heuristic 500.0     | % from Opt 4.76
Opt 498.0       | Net 495.0 | % from Opt 0.60 || Heuristic 409.0     | % from Opt 17.87
Opt 733.0       | Net 692.0 | % from Opt 5.59 || Heuristic 653.0     | % from Opt 10.91
```

# Resource Placement at Edge Devices
**Problem statement**: At each time `t` a randomly sized batch of user's requests arrive, each has its own profile that contains information about the amount of resources (e.g., [`10` units of CPU, `2` units of RAM, `5` units for Memory]) that it needs in order to be processed properly. The incoming requests must be placed at a set of available nodes, each having its own processing capabilities (e.g., [`100` units of CPU, `20` units of RAM, `50` units for Memory]). In real world, these nodes usually are located behind a reverse proxy such as NGNIX, Traefik or Moleculer API Gateway. All of them provide load balancing capabilities. NGNIX [offers](http://nginx.org/en/docs/http/load_balancing.html) round-robin, least-connected, ip-hash; Traefik, at this moment, only [supports](https://docs.traefik.io/routing/services/#load-balancing) round-robin method; Moleculer API Gateway [offers](https://moleculer.services/docs/0.14/balancing.html#Built-in-strategies) round-robin, random, CPU usage-based and sharding. These load balancing strategies don't provide optimal solution, it's too expensive too look for it in real-time, they simply follow the selected load balancing strategy. These strategies are fast but the results that they provide can be suboptimal.

**Goal**: The goal is to design another load balancing strategy that's able to distribute the incoming requests in a fair way, i.e., in a way that the incoming requests have similar working conditions.

**Purpose of the Neural-based load balancing strategy**: A Neural-based load balancing strategy can adapt the distribution policy (heuristic) according to the incoming user's requests and the state of the nodes and, thus, offer a better way of placing the requests.