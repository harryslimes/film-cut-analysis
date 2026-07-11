"""Cut-detector registry. Each entry: name -> callable(video_path) -> DetectResult."""
from . import ffmpeg_scene, pyscenedetect, torch_gpu

REGISTRY = {
    "ffmpeg-cpu":    lambda v: ffmpeg_scene.detect(v, method="cpu"),
    "ffmpeg-cuda":   lambda v: ffmpeg_scene.detect(v, method="cuda"),
    "psd-content":   lambda v: pyscenedetect.detect_content(v),
    "psd-adaptive":  lambda v: pyscenedetect.detect_adaptive(v),
    "torch-cpu":     lambda v: torch_gpu.detect(v, method="cpu"),
    "torch-cuda":    lambda v: torch_gpu.detect(v, method="cuda"),
}

# TransNetV2 is optional (needs the transnetv2-pytorch package + weights).
try:
    from . import transnet
    REGISTRY["transnet-cuda"] = lambda v: transnet.detect(v, method="cuda")
    REGISTRY["transnet-cpu"] = lambda v: transnet.detect(v, method="cpu")
except Exception:
    pass

# Motion-vector detector reads the encoder's own MVs/I-frames (needs PyAV: `pip install av`).
try:
    from . import motion_vectors
    REGISTRY["motion-vectors"] = lambda v: motion_vectors.detect(v)
except Exception:
    pass
