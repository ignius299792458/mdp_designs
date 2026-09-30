Once you have (V_pi(s)) and \(Q_pi(s,a)\), the next step is to distinguish them from the **optimal** functions.

For a known MDP like the Recycling Robot, the Chapter 3 optimality equations are:

$$
V_*(s)=\max_a Q_*(s,a)
$$

and

$$
Q_*(s,a)=\sum_{s'}P(s'|s,a)
\left[R(s,a,s')+\gamma V_*(s')\right]
$$

Then the optimal policy is:

$$
\pi_*(s)\in\arg\max_a Q_*(s,a)
$$

So unlike our previous fixed-policy evaluation, **we no longer average according to the current policy**. We ask: _which action gives the highest expected future return?_

### Continue your notebook with these cells

```python
# Optimal value function

def calculate_V_star(V_star):
    """
    Bellman optimality equation:

    V*(s) = max_a Q*(s, a)
    """
    updated_V_star = {}

    for state in states:
        action_values = []

        for action in actions[state]:
            q_value = 0.0

            for next_state, probability in P[(state, action)].items():
                reward = R[(state, action, next_state)]

                q_value += probability * (
                    reward + gamma * V_star[next_state]
                )

            action_values.append(q_value)

        updated_V_star[state] = max(action_values)

    return updated_V_star
```

Then iteratively apply it:

```python
V_star = {state: 0.0 for state in states}

theta = 1e-6
max_iterations = 1000

for iteration in range(max_iterations):

    updated_V_star = calculate_V_star(V_star)

    delta = max(
        abs(updated_V_star[state] - V_star[state])
        for state in states
    )

    V_star = updated_V_star

    if delta < theta:
        break

print(f"Converged after {iteration + 1} iterations")

for state in states:
    print(f"V*({state}) = {V_star[state]:.6f}")
```

Now calculate the optimal \(Q\):

```python
Q_star = {}

for state in states:
    Q_star[state] = {}

    for action in actions[state]:

        q_value = 0.0

        for next_state, probability in P[(state, action)].items():
            reward = R[(state, action, next_state)]

            q_value += probability * (
                reward + gamma * V_star[next_state]
            )

        Q_star[state][action] = q_value


for state in states:
    print(f"\nState: {state}")

    for action, value in Q_star[state].items():
        print(f"Q*({state}, {action}) = {value:.6f}")
```

Finally, extract the optimal policy:

```python
optimal_policy = {}

for state in states:

    best_action = max(
        Q_star[state],
        key=Q_star[state].get
    )

    optimal_policy[state] = best_action


print("Optimal Policy:")

for state, action in optimal_policy.items():
    print(f"π*({state}) = {action}")
```

### The conceptual difference is important

Your previous code:

```text
fixed π
   ↓
Bellman expectation
   ↓
Vπ
   ↓
Qπ
```

This code:

```text
V*
 ↓
Bellman optimality equation
 ↓
Q*
 ↓
choose max action
 ↓
π*
```

The crucial change is:

$$
\boxed{
V_\pi(s)=\sum_a\pi(a|s)Q_\pi(s,a)
}
$$

versus

$$
\boxed{
V_*(s)=\max_a Q_*(s,a)
}
$$

The first asks:

> **"How good is this particular policy?"**

The second asks:

> **"What is the best achievable expected return from this state?"**

And:

$$
\boxed{\pi_*(s)=\arg\max_a Q_*(s,a)}
$$

gives the corresponding optimal decision.

**One important point:** this is still **planning with a known model**, not learning from robot experience. The robot is not discovering \(P\) or \(R\); we're using the known MDP to calculate the optimum. Actual learning from step-by-step experience comes later when we remove that known-model assumption.
