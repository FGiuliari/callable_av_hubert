import cv2
import tempfile
import argparse
import os
from argparse import Namespace
import fairseq
import avhubert
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.configs import GenerationConfig
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="AV-Hubert Video-to-Text CLI")
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the input cropped video (.mp4)",
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to the model checkpoint (.pt)"
    )
    parser.add_argument(
        "--user_dir",
        type=str,
        default="./asd",
        help="Path to the fairseq user directory",
    )
    parser.add_argument("--beam", type=int, default=20, help="Beam size for generation")
    return parser.parse_args()


def predict(video_path, ckpt_path, user_dir, beam_size=20):
    # 1. Prepare Data Environment
    num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    data_dir = tempfile.mkdtemp()

    # AV-Hubert expects a .tsv file and a label file (.wrd)
    tsv_cont = [
        "/\n",
        f"test-0\t{video_path}\t{None}\t{num_frames}\t{int(16_000*num_frames/25)}\n",
    ]
    label_cont = ["DUMMY\n"]

    with open(os.path.join(data_dir, "test.tsv"), "w") as fo:
        fo.write("".join(tsv_cont))
    with open(os.path.join(data_dir, "test.wrd"), "w") as fo:
        fo.write("".join(label_cont))

    # 2. Setup Fairseq Task
    utils.import_user_module(Namespace(user_dir=user_dir))
    modalities = ["video"]
    gen_subset = "test"

    models, saved_cfg = checkpoint_utils.load_model_ensemble([ckpt_path])
    models = [model.eval().cuda() for model in models]

    saved_cfg.task.modalities = modalities
    saved_cfg.task.data = data_dir
    saved_cfg.task.label_dir = data_dir
    saved_cfg.task.max_sample_size = 600000000

    task = tasks.setup_task(saved_cfg.task)
    task.load_dataset(gen_subset, task_cfg=saved_cfg.task)

    # 3. Setup Generator
    gen_cfg = GenerationConfig(beam=beam_size)
    gen_cfg.max_len_a = 0
    gen_cfg.max_len_b = 30000
    gen_cfg.lm_weight = 0.0
    generator = task.build_generator(models, gen_cfg)

    def decode_fn(x):
        dictionary = task.target_dictionary
        symbols_ignore = generator.symbols_to_strip_from_output
        symbols_ignore.add(dictionary.pad())
        return task.datasets[gen_subset].label_processors[0].decode(x, symbols_ignore)

    # 4. Inference
    itr = task.get_batch_iterator(dataset=task.dataset(gen_subset)).next_epoch_itr(
        shuffle=False
    )
    sample = next(itr)
    sample = utils.move_to_cuda(sample)

    hypos = task.inference_step(generator, models, sample)
    hypo_tokens = hypos[0][0]["tokens"].int().cpu()
    hypo_str = decode_fn(hypo_tokens)

    return hypo_str


def main():
    args = get_args()
    args.video = str(args.video.absolute())

    if not os.path.exists(args.video):
        print(f"Error: Video file not found at {args.video}")
    elif not os.path.exists(args.ckpt):
        print(f"Error: Checkpoint not found at {args.ckpt}")
    else:
        result = predict(args.video, args.ckpt, args.user_dir, args.beam)
        print("-" * 30)
        print(f"Prediction: {result}")
        print("-" * 30)


if __name__ == "__main__":
    main()
