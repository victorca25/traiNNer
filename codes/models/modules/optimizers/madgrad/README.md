## MADGRAD: A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic Optimization

[Paper](https://arxiv.org/abs/2101.11075) | [Project page](https://github.com/facebookresearch/madgrad)

## Getting Started

### Usage

Usage is the same as the official [torch.optim](https://pytorch.org/docs/stable/optim.html) library.

```python
from madgrad import MADGRAD

# define your params
optimizer = MADGRAD(params, lr=1e-2,
                momentum=0.9, eps=1e-6,
                weight_decay=0, decay_type='Adam')
```

## Arguments
For details on `MADGRAD`'s arguments, read the docstring information and the paper.

This version includes [modifications](https://github.com/lessw2020/Best-Deep-Learning-Optimizers/blob/master/madgrad/madgrad_wd.py) to use AdamW style weight decay as an alternative to the Adam style weight decay from the original version. For this, the `decay_type` argument can be used to use either version. Depending on the use case, there are some recommendations to evaluate:
- You may need to use a lower weight decay than you are accustomed to. In many cases down to 0 (no weight decay), if using the `Adam` style weight decay (recommended by Madgrad authors).
- If using the `AdamW` weight decay style, it can be used at same level you would use for regular AdamW optimizer.

In any case, Madgrad is very different than Adam variants, so the recommendation is to test with the defaults, but do a full learning rate sweep as the optimal learning rate will be different from SGD or Adam. Refer to the links above for more information.

## License

MADGRAD is distributed under MIT license.

```
Copyright (c) Facebook, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```

## Citation

```
@misc{defazio2021adaptivity,
      title={Adaptivity without Compromise: A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic Optimization}, 
      author={Aaron Defazio and Samy Jelassi},
      year={2021},
      eprint={2101.11075},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```