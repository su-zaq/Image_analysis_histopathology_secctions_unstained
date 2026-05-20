"""
RGB 画像の各チャンネルでバランス R/(R+G+B), G/(R+G+B), B/(R+G+B) を
計算し、各チャンネルごとにグレースケール画像（8bit）として保存する。

色差 (R−G)/(R+G+B+eps), (G−B)/(R+G+B+eps), (R−B)/(R+G+B+eps) も
rgb_chromatic_diff_grayscale で [−1,1] → [0,255] にマップして利用する。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def rgb_balance_grayscale(
    bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    BGR uint8 画像に対し、R/(R+G+B), G/(...), B(...) を [0,1] で計算し
    8bit グレースケール (0–255) にスケールして返す。
    R+G+B==0 の画素は 0 とする。
    """
    f = bgr.astype(np.float32)
    b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    s = r + g + b
    eps = 1e-6
    s_safe = np.where(s > eps, s, 1.0)  # ゼロ除算回避
    r_bal = r / s_safe
    g_bal = g / s_safe
    b_bal = b / s_safe
    # 和が 0 の画素はバランス不定のため 0
    mask = s > eps
    r_bal = np.where(mask, r_bal, 0.0)
    g_bal = np.where(mask, g_bal, 0.0)
    b_bal = np.where(mask, b_bal, 0.0)
    b_u8 = np.clip(b_bal * 255.0, 0, 255).astype(np.uint8)
    g_u8 = np.clip(g_bal * 255.0, 0, 255).astype(np.uint8)
    r_u8 = np.clip(r_bal * 255.0, 0, 255).astype(np.uint8)
    return b_u8, g_u8, r_u8


def rgb_chromatic_diff_grayscale(
    bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    BGR uint8 に対し
      RG = (R-G)/(R+G+B+eps), GB = (G-B)/(R+G+B+eps), RB = (R-B)/(R+G+B+eps)
    を計算し、値を [-1, 1] にクリップのうえ (v+1)/2 で [0,255] の uint8 にする。

    戻り値は cv2.merge 用で、BGR の並び (B,G,R) = (RB, GB, RG) と対応し、
    BGR→RGB 変換後のテンソルはチャンネル順 (RG, GB, RB) になる。
    """
    eps = 1e-6
    f = bgr.astype(np.float32)
    b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    denom = r + g + b + eps
    rg = (r - g) / denom
    gb = (g - b) / denom
    rb = (r - b) / denom
    rg = np.clip(rg, -1.0, 1.0)
    gb = np.clip(gb, -1.0, 1.0)
    rb = np.clip(rb, -1.0, 1.0)
    rb_u8 = np.clip((rb + 1.0) * 127.5, 0, 255).astype(np.uint8)
    gb_u8 = np.clip((gb + 1.0) * 127.5, 0, 255).astype(np.uint8)
    rg_u8 = np.clip((rg + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return rb_u8, gb_u8, rg_u8


def main() -> None:
    p = argparse.ArgumentParser(
        description="各チャンネル RGB バランス（ch/(R+G+B)）のグレースケールを出力"
    )
    p.add_argument("image", type=Path, help="入力 RGB/BGR 画像のパス")
    p.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help="出力先ディレクトリ（未指定なら画像と同じ場所）",
    )
    args = p.parse_args()

    path = args.image
    if not path.is_file():
        raise SystemExit(f"ファイルが見つかりません: {path}")

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"画像を読み込めませんでした: {path}")

    b_gray, g_gray, r_gray = rgb_balance_grayscale(bgr)

    out_dir = args.out_dir if args.out_dir is not None else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem

    out_b = out_dir / f"{stem}_balance_B.png"
    out_g = out_dir / f"{stem}_balance_G.png"
    out_r = out_dir / f"{stem}_balance_R.png"

    cv2.imwrite(str(out_b), b_gray)
    cv2.imwrite(str(out_g), g_gray)
    cv2.imwrite(str(out_r), r_gray)

    print("保存しました:")
    print(f"  B バランス: {out_b}")
    print(f"  G バランス: {out_g}")
    print(f"  R バランス: {out_r}")


if __name__ == "__main__":
    main()
