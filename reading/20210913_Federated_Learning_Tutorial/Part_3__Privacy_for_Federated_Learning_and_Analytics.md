# Part 3: Privacy for Federated Learning and Analytics

**Original Slides:** [Federated Learning Tutorial](https://sites.google.com/view/fl-tutorial/)

## ML on sensitive data: privacy vs. utility (?)

![img](https://miro.medium.com/max/60/1*9lJgfIe78RUhd8r8JXEIbg.png?q=20)

![img](https://miro.medium.com/max/526/1*9lJgfIe78RUhd8r8JXEIbg.png)

Make achieving high privacy and utility possible with less work.

## What private information might an actor learn

![img](https://miro.medium.com/max/60/1*bvpEuF9brH16ho8JZ6Pv7A.png?q=20)

![img](https://miro.medium.com/max/700/1*bvpEuF9brH16ho8JZ6Pv7A.png)

Privacy principles guiding FL



# Differentially Private Federated Training

**Differential Privacy** (Andy Greenberg Wired 2016.06.13)

> Differential privacy is the statistical science of trying to learn as much as possible about a group while learning as little as possible about any individual in it.

![img](https://miro.medium.com/max/60/1*G-xPfwMzUWmJwB5dok3WNg.png?q=20)

![img](https://miro.medium.com/max/602/1*G-xPfwMzUWmJwB5dok3WNg.png)

**(ε, δ)-Differential Privacy:** The distribution of the output M(D) (a trained model) on the database (training dataset) D is nearly the same as M(D′) for all adjacent databases D and D′.

![img](https://miro.medium.com/max/60/1*gbNJjmBvYaUFcBMUCqdPJw.png?q=20)

![img](https://miro.medium.com/max/391/1*gbNJjmBvYaUFcBMUCqdPJw.png)

H. B. McMahan, et al. Learning Differentially Private Recurrent Language Models. ICLR 2018:

- **Record-level Differential Privacy:** adjacent Sets D and D’ differ only by presence/absence of one example.
- **User-level Differential Privacy:** adjacent Sets D and D’ differ only by presence/absence of one user.

## Iterative training with differential privacy

![img](https://miro.medium.com/max/60/1*8V0otgAvoLr7ioaSo3JGZQ.png?q=20)

![img](https://miro.medium.com/max/700/1*8V0otgAvoLr7ioaSo3JGZQ.png)

1. Sample a batch of clients uniformly at random
2. Clip each update to maximum L2 norm S
3. Average clipped updates
4. Add noise
5. Incorporate into model

![img](https://miro.medium.com/max/60/1*IHC3LEoZ-JEAuOYqbBjSTg.png?q=20)

![img](https://miro.medium.com/max/700/1*IHC3LEoZ-JEAuOYqbBjSTg.png)

![img](https://miro.medium.com/max/60/1*0whk1lKRO8iRGnM7LMbffw.png?q=20)

![img](https://miro.medium.com/max/700/1*0whk1lKRO8iRGnM7LMbffw.png)

## Distributed Differential Privacy

![img](https://miro.medium.com/max/60/1*AY-e0K8ElYXgKq2U7v9Upg.png?q=20)

![img](https://miro.medium.com/max/700/1*AY-e0K8ElYXgKq2U7v9Upg.png)

![img](https://miro.medium.com/max/60/1*BUhMB89Ng1H1AHBRrFhUKw.png?q=20)

![img](https://miro.medium.com/max/700/1*BUhMB89Ng1H1AHBRrFhUKw.png)

Distributed DP via secure aggregation

## Precise DP Guarantees for Real-World Cross-Device FL: **Challenges**

- There is no fixed or known database/dataset/population size
- Client availability is dynamic due to multiple system layers and participation constraints: “Sample from the population” or “shuffle devices” don’t work out-of-the-box
- Clients may drop out at any point of the protocol, with possible impacts on privacy and utility

For privacy purposes, model the environment (availability, dropout) as the choices of Nature (possibly malicious and adaptive to previous mechanism choices)

## Precise DP Guarantees for Real-World Cross-Device FL: **Goals**

- Robust to Nature’s choices (client availability, client dropout) in that privacy and utility are both preserved, possibly at the expense of forward progress.
- Self-accounting, in that the server can compute a precise upper bound on the (ε,δ) of the mechanism using only information available via the protocol.
- Local selection, so most participation decisions are made locally, and as few devices as possible check-in to the server
- Good privacy vs. utility tradeoffs

![img](https://miro.medium.com/max/60/1*CydlzvERp29CFP9PAr03DQ.png?q=20)

![img](https://miro.medium.com/max/700/1*CydlzvERp29CFP9PAr03DQ.png)



# Part IV: Open Problems and Other Topics