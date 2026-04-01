"""Convert wind-downscaling-gan TF discriminator checkpoint to a PyTorch state_dict.

This is intended for the *feature-extractor* use case in Mink-DDPM:
- We use the discriminator *trunk* as a frozen feature extractor (no GAN loss).
- The PyTorch trunk lives in `src/loss.py` as `WindGANDiscriminatorTrunk`.

Usage
-----
1) Ensure you have the TF repo available (or pip-install it) and its deps:
   tensorflow, tensorflow-addons.
2) Run:

   python tools/convert_wind_gan_discriminator_tf_to_torch.py \
     --tf-repo /path/to/wind-downscaling-gan-master \
     --ckpt /path/to/wind-downscaling-gan-master/src/downscaling/weights-55.ckpt \
     --out discriminator_trunk_torch.pth \
     --image-size 128 --timesteps 1 --channels 1

Notes
-----
- This script tries to be robust, but TF spectral-normalization wrappers and
  ConvLSTM variable naming can vary across TF/TF-Addons versions.
- If conversion is not exact, Mink-DDPM will still run, but features will not
  match the original TF model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _as_torch_conv2d_weight(tf_w):
    # TF: [kh, kw, in, out] -> Torch: [out, in, kh, kw]
    w = torch.from_numpy(tf_w)
    return w.permute(3, 2, 0, 1).contiguous()


def _as_torch_bias(tf_b):
    return torch.from_numpy(tf_b).contiguous()


def _pack_convlstm_kernel(kernel, recurrent_kernel, bias, hidden_ch, in_ch):
    """Pack TF ConvLSTM (kernel + recurrent_kernel) into our single-conv weight."""
    # kernel: [kh, kw, in, 4H]
    # recurrent_kernel: [kh, kw, H, 4H]
    # bias: [4H]
    k_t = _as_torch_conv2d_weight(kernel)  # [4H, in, kh, kw]
    rk_t = _as_torch_conv2d_weight(recurrent_kernel)  # [4H, H, kh, kw]
    w = torch.cat([k_t, rk_t], dim=1)  # [4H, in+H, kh, kw]
    b = _as_torch_bias(bias)
    assert w.shape[0] == 4 * hidden_ch
    assert w.shape[1] == in_ch + hidden_ch
    assert b.shape[0] == 4 * hidden_ch
    return w, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf-repo", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--timesteps", type=int, default=1)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--feature-channels", type=int, default=16)
    args = ap.parse_args()

    tf_repo = Path(args.tf_repo).expanduser().resolve()
    ckpt = Path(args.ckpt).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    sys.path.insert(0, str(tf_repo / "src"))

    import numpy as np
    import tensorflow as tf

    # Import TF discriminator builder
    from downscaling.gan.models import make_discriminator

    # Build TF model (expects low_res and high_res same size)
    disc = make_discriminator(
        low_res_size=args.image_size,
        high_res_size=args.image_size,
        low_res_channels=args.channels,
        high_res_channels=args.channels,
        n_timesteps=args.timesteps,
        batch_size=None,
        feature_channels=args.feature_channels,
    )

    # Restore checkpoint
    ckpt_obj = tf.train.Checkpoint(discriminator=disc)
    ckpt_obj.restore(str(ckpt)).expect_partial()

    # Build PyTorch trunk
    from src.loss import WindGANDiscriminatorTrunk

    trunk = WindGANDiscriminatorTrunk(
        in_ch_low=args.channels,
        in_ch_high=args.channels,
        feature_channels=args.feature_channels,
        use_spectral_norm=False,  # weights are for raw convs
    )

    # Force build of dynamic stacks
    dummy = torch.zeros(1, args.channels, args.image_size, args.image_size)
    _ = trunk(dummy)

    sd = trunk.state_dict()

    # Helper: fetch TF weights by layer name
    def get_layer(name: str):
        for l in disc.layers:
            if l.name == name:
                return l
        raise KeyError(f"TF layer not found: {name}")

    # ConvLSTM layers
    # Keras ConvLSTM2D stores weights: [kernel, recurrent_kernel, bias]
    hr_lstm = get_layer("conv_lstm2d")
    mix_lstm = get_layer("conv_lstm2d_1")

    # Our trunk convlstm cells are stored as hr_lstm.cell.conv, etc.
    hr_k, hr_rk, hr_b = hr_lstm.get_weights()
    w, b = _pack_convlstm_kernel(hr_k, hr_rk, hr_b, hidden_ch=args.channels, in_ch=args.channels)
    sd["hr_lstm.cell.conv.weight"] = w
    sd["hr_lstm.cell.conv.bias"] = b

    mix_k, mix_rk, mix_b = mix_lstm.get_weights()
    w, b = _pack_convlstm_kernel(
        mix_k, mix_rk, mix_b,
        hidden_ch=args.feature_channels,
        in_ch=2 * args.channels,
    )
    sd["mix_lstm.cell.conv.weight"] = w
    sd["mix_lstm.cell.conv.bias"] = b

    # First convs after LSTMs are SpectralNormalization-wrapped in TF.
    # We grab the underlying Conv2D weights by searching sublayers.
    def unwrap_sn_conv(layer):
        # TensorFlow Addons SpectralNormalization wraps a layer in `layer.layer`
        if hasattr(layer, "layer"):
            return layer.layer
        return layer

    # hr_conv sits inside a TimeDistributed with SpectralNormalization.
    # Keras naming differs; safest is to search by expected kernel shape.
    def find_conv2d_by_shape(out_ch):
        for l in disc.submodules:
            if isinstance(l, tf.keras.layers.Conv2D):
                k = l.kernel
                if int(k.shape[-1]) == out_ch:
                    return l
        raise RuntimeError("Could not locate Conv2D by shape")

    # This heuristic is imperfect; prefer manual mapping if needed.
    hr_conv_tf = find_conv2d_by_shape(args.feature_channels)
    sd["hr_conv.weight"] = _as_torch_conv2d_weight(hr_conv_tf.get_weights()[0])
    sd["hr_conv.bias"] = _as_torch_bias(hr_conv_tf.get_weights()[1])

    # mix_conv: another conv with out_ch=feature_channels and in_ch=feature_channels
    mix_conv_tf = None
    for l in disc.submodules:
        if isinstance(l, tf.keras.layers.Conv2D):
            w = l.get_weights()[0]
            if w.shape[-1] == args.feature_channels and w.shape[2] == args.feature_channels:
                # skip hr_conv
                if l is not hr_conv_tf:
                    mix_conv_tf = l
                    break
    if mix_conv_tf is None:
        raise RuntimeError("Could not locate mix Conv2D")

    sd["mix_conv.weight"] = _as_torch_conv2d_weight(mix_conv_tf.get_weights()[0])
    sd["mix_conv.bias"] = _as_torch_bias(mix_conv_tf.get_weights()[1])

    # Downsample convs are named conv_{size} in TF.
    # We load them sequentially into our conv lists.
    tf_convs = []
    for l in disc.layers:
        if l.name.startswith("conv_"):
            # TimeDistributed(SpectralNormalization(Conv2D))
            inner = unwrap_sn_conv(l.layer) if hasattr(l, "layer") else l
            # TimeDistributed stores the wrapped layer in `.layer`
            if hasattr(inner, "layer"):
                inner = unwrap_sn_conv(inner)
            # Now expect Conv2D
            if isinstance(inner, tf.keras.layers.Conv2D):
                tf_convs.append(inner)

    # Sort by input size descending (conv_128, conv_42, ...)
    def conv_key(c):
        try:
            return int(c.name.split("_")[-1])
        except Exception:
            return 0

    tf_convs = sorted(tf_convs, key=conv_key, reverse=True)

    # Map to our blocks in build order
    all_pt_convs = list(trunk.down_blocks) + list(trunk.final_blocks)
    n = min(len(tf_convs), len(all_pt_convs))
    for i in range(n):
        w_tf, b_tf = tf_convs[i].get_weights()
        sd[f"{('down_blocks' if i < len(trunk.down_blocks) else 'final_blocks')}.{i if i < len(trunk.down_blocks) else (i-len(trunk.down_blocks))}.weight"] = _as_torch_conv2d_weight(w_tf)
        sd[f"{('down_blocks' if i < len(trunk.down_blocks) else 'final_blocks')}.{i if i < len(trunk.down_blocks) else (i-len(trunk.down_blocks))}.bias"] = _as_torch_bias(b_tf)

    # Save
    trunk.load_state_dict(sd, strict=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trunk.state_dict(), out)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
