## Ranger: a synergistic optimizer combining RAdam (Rectified Adam) LookAhead, and GC (gradient centralization) into one optimizer.

[Project page](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)

## Getting Started

### Usage

Usage is the same as the official [torch.optim](https://pytorch.org/docs/stable/optim.html) library.

```python
from ranger import Ranger

# define your params
optimizer = Ranger(params, lr=1e-3, betas=(.95, 0.999), eps=1e-5,
                 weight_decay=0, alpha=0.5, k=6, N_sma_threshhold=5,
                 use_gc=True, gc_conv_only=False, gc_loc=True)
```

Note: for best training results, make sure you use run with a scheduler with a flat lr for some time and then cosine descent the lr, for example, use a 75% flat lr, then step down and run lower lr for 25%, or cosine descend last 25%. For this purpose, the `FlatCosineDecay` scheduler option available in this repo can be used with a parameter `fixed_niter_rel` equal to 0.75.

## Arguments
`Ranger` shares arguments with [torch.optim.Adam](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam).

There are additional hyperparameters. Check the docstring for more information, as well as the papers used as reference: [Gradient Centralization](https://arxiv.org/abs/2004.01461v2), [RAdam](https://github.com/LiyuanLucasLiu/RAdam), [Lookahead](https://arxiv.org/abs/1907.08610)

## License

Ranger is distributed under the Apache License 2.0.

## Citation

```
@misc{Ranger,
  author = {Wright, Less},
  title = {Ranger - a synergistic optimizer.},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer}}
}
```