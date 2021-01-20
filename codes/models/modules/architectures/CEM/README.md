# Consistency Enforcing Module (CEM)

An architectural module (implemented in PyTorch) that can enforce outputs consistency on **any given super-resolution (SR) model (pre-trained or not):** Wrapping with this module guarantees that its outputs would match the low-resolution inputs, when downsampled using a given (or otherwise the bicubic) downsampling kernel.
<p align="center">
   <img src="fig_CEM_arch.png">
</p>
