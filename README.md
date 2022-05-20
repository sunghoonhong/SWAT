## Structure-Aware Transformer Policy for Inhomogeneous Multi-Task Reinforcement Learning

[Sunghoon Hong](https://sunghoonhong.github.io)

### TL;DR 

Modular Reinforcement Learning, where the agent is assumed to be morphologically structured as a graph, for example composed of limbs and joints, aims to learn a policy that is transferable to a structurally similar but different agent. Compared to traditional Multi-Task Reinforcement Learning, this promising approach allows us to cope with inhomogeneous tasks where the state and action space dimensions differ across tasks. Graph Neural Networks are a natural model for representing the pertinent policies, but a recent work has shown that their multi-hop message passing mechanism is not ideal for conveying important information to other modules and thus a transformer model without morphological information was proposed. In this work, we argue that the morphological information is still very useful and propose a transformer policy model that effectively encodes such information. Specifically, we encode the morphological information in terms of the traversal-based positional embedding and the graph-based relational embedding. We empirically show that the morphological information is crucial for modular reinforcement learning, substantially outperforming prior state-of-the-art methods on multi-task learning as well as transfer learning settings with different state and action space dimensions.


## Setup

All the experiments are done in a Docker container.
To build it, run `./docker_build.sh <device>`, where `<device>` can be `cpu` or `cu101`. It will use CUDA by default.

To build and run the experiments, you need a MuJoCo license. Put it to the root folder before running `docker_build.sh`. 


## Running

```
./docker_run <device_id> # either GPU id or cpu
cd swat             # select the experiment to replicate
python main.py --custom_xml walkers --actor_type swat --critic_type swat --seed 1     # run it on a task
```

## Acknowledgement

- The code is built on top of [Amorpheus](https://github.com/yobibyte/amorpheus) repository.