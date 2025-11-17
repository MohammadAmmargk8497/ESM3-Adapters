import warnings
from typing import Literal

import attr

# This is a circular import, so we can't import SamplingConfig directly.
# from esm.sdk.api import SamplingConfig
# Instead, we'll just check for the attribute.


def validate_sampling_config(
    sampling_config, on_invalid: Literal["raise", "warn"] = "warn"
):
    # Check that all tracks have topk_logprobs less or equal to MAX_TOP_K
    for track in attr.fields(type(sampling_config)):
        track: attr.Attribute
        track_config = getattr(sampling_config, track.name, None)
        if hasattr(track_config, "topk_logprobs"):
            # This is a bit of a hack to get around the circular import.
            # We should probably move the SamplingConfig to a different file.
            if "max_topk" not in track.metadata:
                continue
            max_topk = track.metadata["max_topk"]
            if track_config.topk_logprobs > max_topk:
                msg = (
                    f"Sampling track {track.name} has topk_logprobs={track_config.topk_logprobs} "
                    f"greater than MAX_TOPK={max_topk}."
                )
                if on_invalid == "raise":
                    raise AssertionError(msg)
                else:
                    warnings.warn(msg)
