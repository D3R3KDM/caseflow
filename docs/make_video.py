"""Encode the tour recorded by web/scripts/tour.mjs into docs/media/caseflow_tour.mp4 and a short GIF preview.

    node web/scripts/tour.mjs <dir>          # app must be running on :5190
    python docs/make_video.py <dir>

Uses the ffmpeg binary bundled with the imageio-ffmpeg package (no system install).
"""
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "media"
src = Path(sys.argv[1])
ff = imageio_ffmpeg.get_ffmpeg_exe()
mp4 = OUT / "caseflow_tour.mp4"
gif = OUT / "caseflow_tour.gif"


def run(args):
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", *args], check=True, cwd=src)


run(["-f", "concat", "-safe", "0", "-i", "frames.txt", "-vf", "fps=30,scale=1440:-2",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "slow", "-movflags", "+faststart", str(mp4)])
# GIF preview: the first 13 s (orbit, streamlines, heat slice, worst layout), 12 fps, 860 px wide
run(["-t", "13", "-i", str(mp4), "-vf",
     "fps=12,scale=860:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
     "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", str(gif)])
for f in (mp4, gif):
    print(f.name, f"{f.stat().st_size / 1e6:.1f} MB")
