"""Preprocess a single custom video into rPPG-Toolbox .npy chunks for visualization."""

import argparse
import importlib.util
import os
import sys

from yacs.config import CfgNode as CN

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)


def _load_module(name, rel_path):
    path = os.path.join(REPO_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BaseLoaderModule = _load_module("BaseLoader", "dataset/data_loader/BaseLoader.py")
BaseLoader = BaseLoaderModule.BaseLoader


def read_video(video_file):
    import cv2
    import numpy as np

    vid_obj = cv2.VideoCapture(video_file)
    vid_obj.set(cv2.CAP_PROP_POS_MSEC, 0)
    success, frame = vid_obj.read()
    frames = []
    while success:
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
        frames.append(np.asarray(frame))
        success, frame = vid_obj.read()
    return np.asarray(frames)


def build_preprocess_config(fs=30, chunk_length=180):
    cfg = CN()
    cfg.USE_PSUEDO_PPG_LABEL = True
    cfg.DATA_TYPE = ["DiffNormalized", "Standardized"]
    cfg.DATA_AUG = ["None"]
    cfg.LABEL_TYPE = "DiffNormalized"
    cfg.DO_CHUNK = True
    cfg.CHUNK_LENGTH = chunk_length
    cfg.CROP_FACE = CN()
    cfg.CROP_FACE.DO_CROP_FACE = True
    cfg.CROP_FACE.BACKEND = "HC"
    cfg.CROP_FACE.USE_LARGE_FACE_BOX = True
    cfg.CROP_FACE.LARGE_BOX_COEF = 1.5
    cfg.CROP_FACE.DETECTION = CN()
    cfg.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION = False
    cfg.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY = 30
    cfg.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX = False
    cfg.RESIZE = CN()
    cfg.RESIZE.W = 72
    cfg.RESIZE.H = 72
    return cfg, fs


def build_output_dir(base_dir, chunk_length):
    return os.path.join(
        base_dir,
        "IMG5460_SizeW72_SizeH72_ClipLength{0}_DataTypeDiffNormalized_Standardized_"
        "DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_"
        "Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse".format(
            chunk_length
        ),
    )


class SingleVideoPreprocessor(BaseLoader):
    """Minimal BaseLoader wrapper for one-video preprocessing."""

    def __init__(self, cached_path, fs):
        self.cached_path = cached_path
        self.inputs = []
        self.labels = []
        self.config_data = CN()
        self.config_data.FS = fs


def preprocess_video(video_path, output_dir, fs=30, chunk_length=180, filename="subject1"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Reading video: {video_path}")
    frames = read_video(video_path)
    print(f"Loaded {frames.shape[0]} frames at target fs={fs}")

    config_preprocess, fs = build_preprocess_config(fs=fs, chunk_length=chunk_length)
    loader = SingleVideoPreprocessor(output_dir, fs)
    bvps = loader.generate_pos_psuedo_labels(frames, fs=fs)

    frames_clips, bvps_clips = loader.preprocess(frames, bvps, config_preprocess)
    num_saved = loader.save(frames_clips, bvps_clips, filename)
    print(f"Saved {num_saved} chunk(s) to {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Preprocess a single video for viz_preprocessed_data.ipynb")
    parser.add_argument(
        "--video",
        default="/Users/spencerliang/Documents/Tsinghua-local/IMG_5460.mp4",
        help="Path to input video file",
    )
    parser.add_argument(
        "--output-base",
        default=os.path.join(REPO_ROOT, "preprocessed_data"),
        help="Base directory for preprocessed output",
    )
    parser.add_argument("--fs", type=int, default=30)
    parser.add_argument("--chunk-length", type=int, default=180)
    parser.add_argument("--filename", default="subject1")
    args = parser.parse_args()

    output_dir = build_output_dir(args.output_base, args.chunk_length)
    result_dir = preprocess_video(
        args.video,
        output_dir,
        fs=args.fs,
        chunk_length=args.chunk_length,
        filename=args.filename,
    )
    print(f"\nDone. Point viz_preprocessed_data.ipynb to:\n  {result_dir}")


if __name__ == "__main__":
    main()
