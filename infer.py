"""
推論専用（最初に貼ってくれた experiment.py の save_image / image_compression_save と同じ思想）
- ファイル名は固定しない（INPUT_ROOT 直下の画像を全部対象）
- 入力は cv2 で読み込み → Tensor化（CHW）
- 余計な処理はしない（sigmoid / clamp / 二値化 / min-max 正規化なし）
- 出力は pred * 255 → uint8 → (compress_rate で丸め) → PNG保存

※ 注意：
  あなたの学習コードには assert x.max() <= 1.0 があるので、
  学習時は入力が 0..1 のはずです。
  その場合は推論でも img /= 255.0 を True にする必要があります。
  （これを False にすると真っ黒になりやすい）
"""

import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

from VitLib import create_directory
from VitLib_PyTorch.Network import U_Net, Nested_U_Net
from rgb_balance import rgb_balance_grayscale, rgb_chromatic_diff_grayscale


# =========================
# ★ 設定ここだけ編集 ★
# =========================
MODEL_PATH  = "./result/HE/test02/model/exp0001_test03_epoch20.pth"
INPUT_ROOT  = "./test/bf/x"          # この直下の画像すべてを推論
OUTPUT_ROOT = "./test/bf/y"

DEVICE = "cuda:0"

# 学習時と合わせる
USE_NETWORK = "U-Net"                # "U-Net" or "U-Net++"
EXPERIMENT_SUBJECT = "membrane"      # "membrane" / "nuclear" / "both"
IN_CHANNELS = 3                      # 学習と一致させる。膜・核単独で拡張ch使用時は 6,9,12,18 など
USE_RGB_BALANCE = False              # experiment の use_rgb_balance=True なら True
USE_RGB_CHROMATIC = False            # experiment の use_rgb_chromatic=True なら True（バランスと独立）
INPUT_COLOR = "RGB"                 # 学習の color に合わせる "RGB" or "HSV"
DEEP_SUPERVISION = False             # U-Net++でdeepsupervision使ってたなら True
USE_OTHER_CHANNEL = False            # bothのときのみ（学習と合わせる）
USE_SOFTMAX = False                  # bothのときのみ（学習と合わせる）

# 画像読み込み・前処理（「画像はそのまま」寄り）
BGR_TO_RGB = True                    # 学習がRGBなら True 推奨（値は変わらない）
RESIZE = None                        # 例: (256, 256) / None
INPUT_DIV_255 = True                 # ★学習が 0..1 入力なら True（推奨）
# ↑ 学習時に x.max()<=1 を満たす必要があるなら True にしてください

# 保存時の丸め（最初のコードの compress_rate 相当）
COMPRESS_RATE = 1                    # 1なら丸めなし、2なら2刻み、4なら4刻み…


# =========================
# 画像拡張子判定
# =========================
def is_image_file(name: str) -> bool:
    return name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"))


# =========================
# モデル構築（学習コードの train_roop と同じ出力chに合わせる）
# =========================
def build_model() -> torch.nn.Module:
    if USE_NETWORK == "U-Net":
        if EXPERIMENT_SUBJECT == "both":
            out_ch = 3 if USE_OTHER_CHANNEL else 2
            return U_Net(IN_CHANNELS, out_ch, bilinear=False, softmax=USE_SOFTMAX)
        return U_Net(IN_CHANNELS, 1, bilinear=False)

    if USE_NETWORK == "U-Net++":
        if EXPERIMENT_SUBJECT == "both":
            out_ch = 3 if USE_OTHER_CHANNEL else 2
            return Nested_U_Net(
                IN_CHANNELS,
                out_ch,
                softmax=USE_SOFTMAX,
                deepsupervision=DEEP_SUPERVISION
            )
        return Nested_U_Net(
            IN_CHANNELS,
            1,
            deepsupervision=DEEP_SUPERVISION
        )

    raise ValueError("USE_NETWORK must be 'U-Net' or 'U-Net++'")


# =========================
# 画像読み込み（ファイル名固定なし）
# =========================
def load_image_tensor(path: str) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")

    # grayscale -> HWC
    if img.ndim == 2:
        img = img[:, :, None]

    if (
        (USE_RGB_BALANCE or USE_RGB_CHROMATIC)
        and EXPERIMENT_SUBJECT in ("membrane", "nuclear")
        and img.shape[2] >= 3
    ):
        bgr = img[:, :, :3].astype(np.uint8)
        if INPUT_COLOR == "RGB":
            p1 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        else:
            p1 = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        if INPUT_DIV_255:
            p1 /= 255.0
        parts = [p1]
        if USE_RGB_BALANCE:
            b0, g0, r0 = rgb_balance_grayscale(bgr)
            bal_bgr = cv2.merge([b0, g0, r0])
            if INPUT_COLOR == "RGB":
                pb = cv2.cvtColor(bal_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            else:
                pb = cv2.cvtColor(bal_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            if INPUT_DIV_255:
                pb /= 255.0
            parts.append(pb)
        if USE_RGB_CHROMATIC:
            rb, gb, rg = rgb_chromatic_diff_grayscale(bgr)
            cd_bgr = cv2.merge([rb, gb, rg])
            if INPUT_COLOR == "RGB":
                pc = cv2.cvtColor(cd_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            else:
                pc = cv2.cvtColor(cd_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            if INPUT_DIV_255:
                pc /= 255.0
            parts.append(pc)
        img = np.concatenate(parts, axis=2)
        if RESIZE is not None:
            img = cv2.resize(img, RESIZE, interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(img).permute(2, 0, 1)
        if x.shape[0] < IN_CHANNELS:
            pad = torch.zeros(
                (IN_CHANNELS - x.shape[0], x.shape[1], x.shape[2]), dtype=x.dtype
            )
            x = torch.cat([x, pad], dim=0)
        elif x.shape[0] > IN_CHANNELS:
            x = x[:IN_CHANNELS]
        return x

    # チャンネル数合わせ
    if img.shape[2] >= 3 and IN_CHANNELS >= 3:
        img = img[:, :, :3]
        if BGR_TO_RGB:
            img = img[:, :, ::-1]  # BGR -> RGB（値は変わらない）
    else:
        img = img[:, :, :IN_CHANNELS]

    # resize（必要なときだけ）
    if RESIZE is not None:
        img = cv2.resize(img, RESIZE, interpolation=cv2.INTER_LINEAR)

    # float化
    img = img.astype(np.float32)

    # 入力のスケール（学習と一致させる）
    if INPUT_DIV_255:
        # 16bitの可能性があるなら 65535 にしたい場合はここを調整
        img /= 255.0

    # HWC -> CHW
    x = torch.from_numpy(img).permute(2, 0, 1)
    if x.shape[0] < IN_CHANNELS:
        pad = torch.zeros(
            (IN_CHANNELS - x.shape[0], x.shape[1], x.shape[2]), dtype=x.dtype
        )
        x = torch.cat([x, pad], dim=0)
    elif x.shape[0] > IN_CHANNELS:
        x = x[:IN_CHANNELS]
    return x


# =========================
# 最初のコードと同等：pred * 255 -> uint8 -> (compress_rate) -> 保存
# =========================
def image_compression_save(
    pred: torch.Tensor,
    path: str,
    index: int = 0,
    divide: int = 1,
    channel: int = 0,
) -> None:
    img_torch = pred[index][channel]  # [H,W]
    img_cv2 = np.array(img_torch.detach().cpu().numpy() * 255, dtype=np.uint8)

    if divide > 1:
        img_cv2 = np.where(img_cv2 % divide == 0, img_cv2, img_cv2 - img_cv2 % divide)

    Image.fromarray(img_cv2, mode="L").save(path)


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # 出力先作成
    create_directory(OUTPUT_ROOT)

    # モデル作成 & 重みロード
    model = build_model().to(device)

    # 未来警告回避（PyTorchが対応していれば weights_only=True）
    try:
        ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(MODEL_PATH, map_location=device)

    state_dict = ckpt.get("state_dict", ckpt)

    # DataParallel "module." 対応
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=True)

    model.eval()

    # INPUT_ROOT 直下の画像のみ
    image_files = [
        f for f in os.listdir(INPUT_ROOT)
        if os.path.isfile(os.path.join(INPUT_ROOT, f)) and is_image_file(f)
    ]

    for fname in tqdm(image_files, desc="predict"):
        in_path = os.path.join(INPUT_ROOT, fname)

        # 入力テンソル
        x = load_image_tensor(in_path).unsqueeze(0).to(device)  # [1,C,H,W]

        with torch.no_grad():
            pred = model(x)
            if isinstance(pred, list):  # U-Net++ deep supervision
                pred = pred[-1]

        # 保存（最初のコードと同じ方式）
        base, ext = os.path.splitext(fname)

        if EXPERIMENT_SUBJECT == "both":
            out_mem = os.path.join(OUTPUT_ROOT, f"{base}_mem.png")
            out_nuc = os.path.join(OUTPUT_ROOT, f"{base}_nuc.png")
            image_compression_save(pred, out_mem, divide=COMPRESS_RATE, channel=0)
            image_compression_save(pred, out_nuc, divide=COMPRESS_RATE, channel=1)
        else:
            out_path = os.path.join(OUTPUT_ROOT, f"{base}.png")
            image_compression_save(pred, out_path, divide=COMPRESS_RATE, channel=0)

        # メモリ掃除（大量画像なら有効）
        del x, pred
        torch.cuda.empty_cache()

    print("=== prediction finished ===")


if __name__ == "__main__":
    main()
