def rollout(self, acts):
    """
    Perform a rollout using a preset collection of actions
    """
    total_reward = 0
    self.env.reset()
    steps = 0
    for act in acts:
        if (self.render):
            self.env.render()
        obs, reward, done, info = self.env.step(act)
        steps += 1
        total_reward += reward
        if done:
            break

    return steps, total_reward


def update_tree(self, executed_acts, total_reward):
    """
    Given the tree, a list of actions that were executed before the game ended, and a reward, update the tree
    so that the path formed by the executed actions are all updated to the new reward.
    """
    self.root.value = max(total_reward, self.root.value)
    self.root.visits += 1
    new_nodes = 0

    node = self.root
    for step, act in enumerate(executed_acts):
        if act not in node.children:
            node.children[act] = Node.Node()
            new_nodes += 1
        node = node.children[act]
        node.value = max(total_reward, node.value)
        node.visits += 1

    return new_nodes
