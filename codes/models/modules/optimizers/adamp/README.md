## AdamP: Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights (ICLR 2021)

[Paper](https://arxiv.org/abs/2006.08217) | [Project page](https://clovaai.github.io/AdamP/)

## How does it work?

Please visit the [project page](https://clovaai.github.io/AdamP/).

## Getting Started

### Usage

Usage is the same as the official [torch.optim](https://pytorch.org/docs/stable/optim.html) library.

```python
from adamp import AdamP

# define your params
optimizer = AdamP(params, lr=0.001, betas=(0.9, 0.999), weight_decay=1e-2)
```

```python
from adamp import SGDP

# define your params
optimizer = SGDP(params, lr=0.1, weight_decay=1e-5, momentum=0.9, nesterov=True)
```

## Arguments
`SGDP` and `AdamP` share arguments with [torch.optim.SGD](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD) and [torch.optim.Adam](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam).
There are two additional hyperparameters; we recommend using the default values.
- `delta` : threhold that determines whether a set of parameters is scale invariant or not (default: 0.1)
- `wd_ratio` : relative weight decay applied on _scale-invariant_ parameters compared to that applied on _scale-variant_ parameters (default: 0.1)

Both `SGDP` and `AdamP` support Nesterov momentum.
- `nesterov` : enables Nesterov momentum (default: False)

## License

AdamP is distributed under MIT license.

```
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## Citation

```
@inproceedings{heo2021adamp,
    title={AdamP: Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights},
    author={Heo, Byeongho and Chun, Sanghyuk and Oh, Seong Joon and Han, Dongyoon and Yun, Sangdoo and Kim, Gyuwan and Uh, Youngjung and Ha, Jung-Woo},
    year={2021},
    booktitle={International Conference on Learning Representations (ICLR)},
}
```