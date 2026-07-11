"""Render a contact sheet of every detected cut in a timespan: the frame just-before and
just-after each cut, side by side, labelled with the timestamp. Lets a human eyeball which
detections are real cuts vs flashes -- ground truth we can then measure against.

  python contact_sheet.py --video <mkv> --start 5030 --end 5115 --out results/vertigo_night.png
"""
import argparse, json, subprocess
import numpy as np
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--cuts", required=True, help="cuts.json with detected cut times")
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--fps", type=float, default=23.976)
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta", type=float, default=0.2, help="seconds before/after cut")
    ap.add_argument("--tw", type=int, default=160)
    ap.add_argument("--cols", type=int, default=4, help="cut-pairs per row")
    args = ap.parse_args()

    a = args.start - 1.0
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(a),
           "-t", str(args.end - a + 1.0), "-i", args.video,
           "-vf", f"scale={args.tw}:-1", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    # infer height from a probe
    probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0", args.video]).decode()
    W, H = map(int, probe.strip().split(","))
    th = int(round(args.tw * H / W / 2) * 2)
    F = np.frombuffer(raw, np.uint8).reshape([-1, th, args.tw, 3])

    cuts = [c for c in json.load(open(args.cuts))["cuts"] if args.start <= c <= args.end]
    d = int(round(args.delta * args.fps))
    pad, lab_h = 6, 18
    pair_w = args.tw * 2 + pad
    cell_w = pair_w + pad
    cell_h = th + lab_h + pad
    cols = args.cols
    rows = (len(cuts) + cols - 1) // cols
    sheet = np.full((rows * cell_h + pad, cols * cell_w + pad, 3), 30, np.uint8)

    for idx, c in enumerate(cuts):
        i = int(round((c - a) * args.fps))
        b = F[max(0, i - d)]
        af = F[min(len(F) - 1, i + d)]
        r, cc = divmod(idx, cols)
        y = pad + r * cell_h + lab_h
        x = pad + cc * cell_w
        sheet[y:y + th, x:x + args.tw] = b
        sheet[y:y + th, x + args.tw + pad:x + 2 * args.tw + pad] = af
        cv2.putText(sheet, f"#{idx+1} {c:.1f}s", (x, pad + r * cell_h + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite(args.out, sheet)
    print(f"{len(cuts)} cut-pairs -> {args.out}  ({sheet.shape[1]}x{sheet.shape[0]})")
    print("left=before, right=after each detected cut. same image both sides => flash/false.")


if __name__ == "__main__":
    main()
