# General description

We know _a priori_ the `capacities` of the backpacks. What de don't know is the `size` and the `value` of the items that we want to pack into the backpacks.

By modeling the knapsack problem this way it becomes similar to the problem that I need to solve.

I have a set of nodes located at `x` and `y` and they have an array of features [`CPU`,`RAM`,`MEM`] that characterizes them. What I don't know is the user's demands. Each demand has its own array of features [`CPU`,`RAM`,`MEM`] describing how much resources they need to be satisfied.

The goal in my problem is to assign/place the demands to nodes in a best possible way.