import pathlib
from typing import Optional, Tuple

import torch
from torch.utils import data
import torchvision
from PIL import Image
import cv2
import numpy as np

from rgb_balance import (
    rgb_balance_grayscale,
    rgb_chromatic_diff_grayscale,
    append_rgb_derived_planes_to_core,
    planes_from_bgr_modal,
)

class Dataset(data.Dataset):
    def __init__(
        self,
        folder_path,
        use_list,
        color='RGB',
        blend='concatenate',
        use_rgb_balance: bool = False,
        use_rgb_chromatic: bool = False,
        use_rgb_core: bool = True,
        use_balance_input_only: bool = False,
        modality_folder_names: Optional[Tuple[str, ...]] = None,
    ):
        """
        Args:
            folder_path (list): 画像フォルダのパス
            use_list (list): 使用する画像のリスト
                - 0: 使用しない
                - 1: 使用する
            color (str): 色空間
                - 'RGB': RGB
                - 'HSV': HSV
            blend (str): 画像の結合方法
                - 'concatenate': 連結
                - 'alpha': αブレンド
        """
        self.modality_folder_names = modality_folder_names
        self.mod_img_paths: list[list[str]] = []
        self.bf_img_paths: list[str] = []
        self.df_img_paths: list[str] = []
        self.ph_img_paths: list[str] = []
        self.y_img_paths: list[str] = []
        self.bf_bal_img_paths: list[str] = []
        self.df_bal_img_paths: list[str] = []
        self.ph_bal_img_paths: list[str] = []
        self.bf_cdiff_img_paths: list[str] = []
        self.df_cdiff_img_paths: list[str] = []
        self.ph_cdiff_img_paths: list[str] = []
        self.use_rgb_balance = use_rgb_balance
        self.use_rgb_chromatic = use_rgb_chromatic
        self.use_rgb_core = True if not (use_rgb_balance or use_rgb_chromatic) else use_rgb_core
        self.use_balance_input_only = use_balance_input_only
        if use_balance_input_only and (use_rgb_balance or use_rgb_chromatic):
            raise Exception('use_balance_input_only 時は use_rgb_balance / use_rgb_chromatic と併用できません。')
        if modality_folder_names is not None:
            if use_balance_input_only:
                raise Exception('modality_folder_names と use_balance_input_only は併用できません。')
            self.mod_img_paths = [[] for _ in modality_folder_names]
            for raw in folder_path:
                root = pathlib.Path(raw).expanduser()
                try:
                    root = root.resolve()
                except OSError:
                    pass
                for i, name in enumerate(modality_folder_names):
                    self.mod_img_paths[i] += self._get_file_path(root / name)
                self.y_img_paths += self._get_file_path(root / 'y')
            n0 = len(self.mod_img_paths[0]) if self.mod_img_paths else 0
            if n0 == 0:
                raise ValueError(
                    '学習用画像が0枚です（無染色6入力: 各 modality サブフォルダを確認）。'
                    f' folder_path={folder_path!r}'
                )
            for i in range(len(modality_folder_names)):
                if len(self.mod_img_paths[i]) != n0:
                    raise ValueError(
                        f'無染色6入力: モダリティ間で枚数が一致しません slot0={n0}, slot{i}={len(self.mod_img_paths[i])}'
                    )
            if len(self.y_img_paths) != n0:
                raise ValueError(
                    f'無染色6入力: y と入力画像の枚数が一致しません y={len(self.y_img_paths)}, inputs={n0}'
                    f' folder_path={folder_path!r}'
                )
        else:
            for raw in folder_path:
                root = pathlib.Path(raw).expanduser()
                try:
                    root = root.resolve()
                except OSError:
                    pass
                if use_balance_input_only:
                    self.bf_bal_img_paths += self._get_file_path(root / 'bf_bal')
                    self.df_bal_img_paths += self._get_file_path(root / 'df_bal')
                    self.ph_bal_img_paths += self._get_file_path(root / 'ph_bal')
                    self.y_img_paths += self._get_file_path(root / 'y')
                else:
                    self.bf_img_paths += self._get_file_path(root / 'bf')
                    self.df_img_paths += self._get_file_path(root / 'df')
                    self.ph_img_paths += self._get_file_path(root / 'ph')
                    if use_rgb_balance and self.use_rgb_core:
                        self.bf_bal_img_paths += self._get_file_path(root / 'bf_bal')
                        self.df_bal_img_paths += self._get_file_path(root / 'df_bal')
                        self.ph_bal_img_paths += self._get_file_path(root / 'ph_bal')
                    if use_rgb_chromatic and self.use_rgb_core:
                        self.bf_cdiff_img_paths += self._get_file_path(root / 'bf_cdiff')
                        self.df_cdiff_img_paths += self._get_file_path(root / 'df_cdiff')
                        self.ph_cdiff_img_paths += self._get_file_path(root / 'ph_cdiff')
                    self.y_img_paths += self._get_file_path(root / 'y')
        if use_balance_input_only:
            n = len(self.y_img_paths)
            if n == 0:
                diag = self._diagnose_balance_folders(folder_path)
                raise ValueError(
                    '学習用データが0枚です（y/ が空、または学習セットのパスに到達できていません）。\n'
                    'membrane_balance / nuclear_balance では各 train_data 配下に bf_bal, df_bal, ph_bal, y が必要です。\n'
                    f'【診断】\n{diag}\n'
                    f'folder_path={folder_path!r}\n'
                    '対処: start_num=0 で拡張画像を生成し直す（同一 default_path に log/exp.log があるとブロックされる場合は別出力先にするか削除）。'
                )
            if len(self.bf_bal_img_paths) != n or len(self.df_bal_img_paths) != n or len(self.ph_bal_img_paths) != n:
                raise ValueError(
                    f'y と *_bal の枚数が一致しません (y={n}, bf_bal={len(self.bf_bal_img_paths)}, '
                    f'df_bal={len(self.df_bal_img_paths)}, ph_bal={len(self.ph_bal_img_paths)})。folder_path={folder_path!r}'
                )
        elif modality_folder_names is None and len(self.bf_img_paths) == 0:
            raise ValueError(
                '学習用画像が0枚です（いずれのフォルダの bf/ にもファイルがありません）。'
                f' folder_path={folder_path!r}。'
                'default_path/train_data 以下へデータ拡張が完了しているか確認してください。'
            )
        self.use_list = use_list
        self.color = color
        self.blend = blend
        self.to_tensor = torchvision.transforms.ToTensor()

        if modality_folder_names is not None and (blend != 'concatenate' or len(use_list) != 6):
            raise Exception('無染色6入力(modality_folder_names)では blend=concatenate かつ len(use_list)==6 が必要です。')
        if use_balance_input_only and modality_folder_names is None and (blend != 'concatenate' or len(use_list) != 3):
            raise Exception('use_balance_input_only 時は blend=concatenate かつ use_list の長さ 3 である必要があります。')
        if modality_folder_names is None and (use_rgb_balance or use_rgb_chromatic) and (blend != 'concatenate' or len(use_list) != 3):
            raise Exception('use_rgb_balance / use_rgb_chromatic 時は blend=concatenate かつ use_list の長さ 3 である必要があります。')
        if (
            not use_balance_input_only
            and modality_folder_names is None
            and use_rgb_balance
            and self.use_rgb_core
            and len(self.bf_img_paths) != len(self.bf_bal_img_paths)
        ):
            raise Exception('bf と bf_bal の枚数が一致しません。')
        if (
            not use_balance_input_only
            and modality_folder_names is None
            and use_rgb_chromatic
            and self.use_rgb_core
            and len(self.bf_img_paths) != len(self.bf_cdiff_img_paths)
        ):
            raise Exception('bf と bf_cdiff の枚数が一致しません。')

        if blend == 'alpha' and len(use_list)!=3:
            raise Exception(f'Blend mode "alpha" is only available when use_list length is 3: {len(use_list)}')
        if blend == 'alpha' and sum(use_list)!=1:
            # αブレンディングの場合は、use_listの合計が1である必要があります。
            raise Exception(f'Blend mode "alpha" is only available when sum of use_list is 1: {sum(use_list)}')

    def _read_modality_plane(self, paths: list, index: int) -> np.ndarray:
        img = cv2.imread(paths[index], cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f'画像の読み込みに失敗しました index={index} path={paths[index]}')
        if self.color == 'RGB':
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.color == 'HSV':
            return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        raise Exception(f'Invalid color: {self.color}')

    def _read_modality_bundle(self, paths: list, index: int) -> np.ndarray:
        """無染色用: モダリティ1枚ぶん RGB/HSV + オンザフライの色差(・バランス)をチャンネル方向に結合"""
        bgr = cv2.imread(paths[index], cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f'画像の読み込みに失敗しました index={index} path={paths[index]}')
        return planes_from_bgr_modal(
            bgr, self.color, self.use_rgb_balance, self.use_rgb_chromatic,
            use_rgb_core=self.use_rgb_core,
        )

    def _append_rgb_extra_channels(self, base_paths, index):
        bgr = cv2.imread(base_paths[index], cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f'画像の読み込みに失敗しました path={base_paths[index]}')
        if not self.use_rgb_core:
            return planes_from_bgr_modal(
                bgr, self.color, self.use_rgb_balance, self.use_rgb_chromatic, use_rgb_core=False,
            )
        im = self._get_image(base_paths, index, self.color)
        if base_paths is self.bf_img_paths:
            bal_paths = self.bf_bal_img_paths
            cdiff_paths = self.bf_cdiff_img_paths
        elif base_paths is self.df_img_paths:
            bal_paths = self.df_bal_img_paths
            cdiff_paths = self.df_cdiff_img_paths
        else:
            bal_paths = self.ph_bal_img_paths
            cdiff_paths = self.ph_cdiff_img_paths
        if self.use_rgb_balance:
            im_bal = self._get_image(bal_paths, index, self.color)
            im = np.concatenate([im, im_bal], axis=2)
        if self.use_rgb_chromatic:
            im_cdiff = self._get_image(cdiff_paths, index, self.color)
            im = np.concatenate([im, im_cdiff], axis=2)
        return im

    def __getitem__(self, index):
        if self.blend == 'concatenate':
            img_list = []
            if len(self.use_list)==3:#撮像法のみの検討
                if self.use_balance_input_only:
                    if self.use_list[0]==1:
                        img_list.append(self._get_image(self.bf_bal_img_paths, index, self.color))
                    if self.use_list[1]==1:
                        img_list.append(self._get_image(self.df_bal_img_paths, index, self.color))
                    if self.use_list[2]==1:
                        img_list.append(self._get_image(self.ph_bal_img_paths, index, self.color))
                    x = np.concatenate(img_list, axis=2)
                else:
                    if self.use_list[0]==1:
                        img_list.append(self._append_rgb_extra_channels(self.bf_img_paths, index))
                    if self.use_list[1]==1:
                        img_list.append(self._append_rgb_extra_channels(self.df_img_paths, index))
                    if self.use_list[2]==1:
                        img_list.append(self._append_rgb_extra_channels(self.ph_img_paths, index))
                    x = np.concatenate(img_list, axis=2) if (self.use_rgb_balance or self.use_rgb_chromatic) else cv2.merge(img_list)
            elif len(self.use_list) == 6 and self.modality_folder_names:
                blocks = []
                for slot in range(6):
                    if self.use_list[slot] == 1:
                        blocks.append(self._read_modality_bundle(self.mod_img_paths[slot], index))
                x = np.concatenate(blocks, axis=2)
            elif len(self.use_list)==9:#色空間毎の検討
                if 1 in self.use_list[0:3]:
                    img_list.append(self._get_image(self.bf_img_paths, index, self.color, self.use_list[0:3]))
                if 1 in self.use_list[3:6]:
                    img_list.append(self._get_image(self.df_img_paths, index, self.color, self.use_list[3:6]))
                if 1 in self.use_list[6:9]:
                    img_list.append(self._get_image(self.ph_img_paths, index, self.color, self.use_list[6:9]))
                x = cv2.merge(img_list)
            elif len(self.use_list)==18:#色空間毎の検討
                if 1 in self.use_list[0:3]:
                    img_list.append(self._get_image(self.bf_img_paths, index, 'RGB', self.use_list[0:3]))
                if 1 in self.use_list[3:6]:
                    img_list.append(self._get_image(self.df_img_paths, index, 'RGB', self.use_list[3:6]))
                if 1 in self.use_list[6:9]:
                    img_list.append(self._get_image(self.ph_img_paths, index, 'RGB', self.use_list[6:9]))
                if 1 in self.use_list[9:12]:
                    img_list.append(self._get_image(self.bf_img_paths, index, 'HSV', self.use_list[9:12]))
                if 1 in self.use_list[12:15]:
                    img_list.append(self._get_image(self.df_img_paths, index, 'HSV', self.use_list[12:15]))
                if 1 in self.use_list[15:18]:
                    img_list.append(self._get_image(self.ph_img_paths, index, 'HSV', self.use_list[15:18]))
                x = cv2.merge(img_list)
            y = Image.open(self.y_img_paths[index]).convert('L')
            return self.to_tensor(x),self.to_tensor(y)
        elif self.blend == 'alpha':
            bf = cv2.imread(self.bf_img_paths[index],cv2.IMREAD_COLOR)
            df = cv2.imread(self.df_img_paths[index],cv2.IMREAD_COLOR)
            ph = cv2.imread(self.ph_img_paths[index],cv2.IMREAD_COLOR)
            y = Image.open(self.y_img_paths[index]).convert('L')
            if self.color == 'RGB':
                bf = cv2.cvtColor(bf, cv2.COLOR_BGR2RGB)
                df = cv2.cvtColor(df, cv2.COLOR_BGR2RGB)
                ph = cv2.cvtColor(ph, cv2.COLOR_BGR2RGB)
            elif self.color == 'HSV':
                bf = cv2.cvtColor(bf, cv2.COLOR_BGR2HSV)
                df = cv2.cvtColor(df, cv2.COLOR_BGR2HSV)
                ph = cv2.cvtColor(ph, cv2.COLOR_BGR2HSV)
            x = bf * self.use_list[0] + df * self.use_list[1] + ph * self.use_list[2]
            x = x.astype(np.uint8)
            return self.to_tensor(x),self.to_tensor(y)
        else:
            raise Exception(f'Invalid blend mode: {self.blend}')

    def _get_image(self, img_path_list, index, color, use_list=None):
        img = cv2.imread(img_path_list[index],cv2.IMREAD_COLOR)
        if color == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif color == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if len(self.use_list)==3:
            return img
        else:
            img_list = []
            r, g, b = cv2.split(img)
            if use_list[0]==1:
                img_list.append(r)
            if use_list[1]==1:
                img_list.append(g)
            if use_list[2]==1:
                img_list.append(b)
            return cv2.merge(img_list)

    def __len__(self):
        if self.modality_folder_names:
            return len(self.mod_img_paths[0])
        if self.use_balance_input_only:
            return len(self.y_img_paths)
        return len(self.bf_img_paths)

    def _diagnose_balance_folders(self, folder_path) -> str:
        lines = []
        for raw in folder_path:
            root = pathlib.Path(raw).expanduser()
            try:
                root = root.resolve()
            except OSError:
                pass
            lines.append(f'セット: {raw} -> resolve: {root}')
            if not root.is_dir():
                lines.append('             [このパスにディレクトリがありません。カレントディレクトリからの相対パスを確認してください]')
                continue
            for name in ('y', 'bf_bal', 'df_bal', 'ph_bal'):
                d = root / name
                if not d.is_dir():
                    lines.append(f'             {name}/ : (なし)')
                else:
                    n = sum(1 for x in d.iterdir() if x.is_file())
                    lines.append(f'             {name}/ : {n} ファイル')
        return '\n'.join(lines)

    def _get_file_path(self, dir_path) -> list:
        base = pathlib.Path(dir_path).expanduser()
        try:
            base = base.resolve()
        except OSError:
            pass
        if not base.is_dir():
            return []
        files = sorted((p for p in base.iterdir() if p.is_file()), key=lambda p: p.name)
        return [str(p) for p in files]

def get_dataloader(folder_path, use_list, color='RGB', blend='concatenate', batch_size = 1, num_workers=0, isShuffle=True, pin_memory=True, use_rgb_balance: bool = False, use_rgb_chromatic: bool = False, use_rgb_core: bool = True, use_balance_input_only: bool = False, modality_folder_names: Optional[Tuple[str, ...]] = None):
    dataset = Dataset(folder_path, use_list, color=color, blend=blend, use_rgb_balance=use_rgb_balance, use_rgb_chromatic=use_rgb_chromatic, use_rgb_core=use_rgb_core, use_balance_input_only=use_balance_input_only, modality_folder_names=modality_folder_names)
    return data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=isShuffle, pin_memory=pin_memory)

def _get_image(img_path, color, use_list=None):
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        if color == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif color == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if use_list is None:
            return img
        else:
            img_list = []
            r, g, b = cv2.split(img)
            if use_list[0]==1:
                img_list.append(r)
            if use_list[1]==1:
                img_list.append(g)
            if use_list[2]==1:
                img_list.append(b)
            return cv2.merge(img_list)

def _apply_rgb_extras_numpy(
    im: np.ndarray,
    bgr: np.ndarray,
    color: str,
    use_rgb_balance: bool,
    use_rgb_chromatic: bool,
    use_rgb_core: bool = True,
) -> np.ndarray:
    if not use_rgb_core:
        return planes_from_bgr_modal(
            np.asarray(bgr, dtype=np.uint8), color, use_rgb_balance, use_rgb_chromatic, use_rgb_core=False,
        )
    return append_rgb_derived_planes_to_core(im, bgr, color, use_rgb_balance, use_rgb_chromatic)

def _balance_only_from_bgr(bgr: np.ndarray, color: str) -> np.ndarray:
    bb, gg, rr = rgb_balance_grayscale(bgr)
    bal_bgr = cv2.merge([bb, gg, rr])
    if color == 'RGB':
        return cv2.cvtColor(bal_bgr, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(bal_bgr, cv2.COLOR_BGR2HSV)


def get_image(img_path_list, use_list, color='RGB', blend='concatenate', use_rgb_balance: bool = False, use_rgb_chromatic: bool = False, use_rgb_core: bool = True, use_balance_input_only: bool = False):
    if blend == 'concatenate':
        img_list = []
        if len(use_list)==3:#撮像法のみの検討
            if use_balance_input_only:
                if use_list[0]==1:
                    bgr0 = cv2.imread(img_path_list[0], cv2.IMREAD_COLOR)
                    img_list.append(_balance_only_from_bgr(bgr0, color))
                if use_list[1]==1:
                    bgr1 = cv2.imread(img_path_list[1], cv2.IMREAD_COLOR)
                    img_list.append(_balance_only_from_bgr(bgr1, color))
                if use_list[2]==1:
                    bgr2 = cv2.imread(img_path_list[2], cv2.IMREAD_COLOR)
                    img_list.append(_balance_only_from_bgr(bgr2, color))
                img = np.concatenate(img_list, axis=2)
                img = torchvision.transforms.ToTensor()(img)
                img = torch.reshape(img, (-1, *img.size()))
                return img
            _uc = True if not (use_rgb_balance or use_rgb_chromatic) else use_rgb_core

            if use_list[0]==1:
                if use_rgb_balance or use_rgb_chromatic:
                    bgr = cv2.imread(img_path_list[0], cv2.IMREAD_COLOR)
                    im0 = _apply_rgb_extras_numpy(None, bgr, color, use_rgb_balance, use_rgb_chromatic, use_rgb_core=_uc)
                else:
                    im0 = _get_image(img_path_list[0], color)
                img_list.append(im0)
            if use_list[1]==1:
                if use_rgb_balance or use_rgb_chromatic:
                    bgr = cv2.imread(img_path_list[1], cv2.IMREAD_COLOR)
                    im1 = _apply_rgb_extras_numpy(None, bgr, color, use_rgb_balance, use_rgb_chromatic, use_rgb_core=_uc)
                else:
                    im1 = _get_image(img_path_list[1], color)
                img_list.append(im1)
            if use_list[2]==1:
                if use_rgb_balance or use_rgb_chromatic:
                    bgr = cv2.imread(img_path_list[2], cv2.IMREAD_COLOR)
                    im2 = _apply_rgb_extras_numpy(None, bgr, color, use_rgb_balance, use_rgb_chromatic, use_rgb_core=_uc)
                else:
                    im2 = _get_image(img_path_list[2], color)
                img_list.append(im2)
            img = np.concatenate(img_list, axis=2) if (use_rgb_balance or use_rgb_chromatic) else cv2.merge(img_list)
        elif len(use_list) == 6:
            if use_balance_input_only:
                raise Exception('無染色6入力の get_image は balance_only と併用できません。')
            if len(img_path_list) < 6:
                raise Exception(f'無染色6入力では img_path_list に6経路が必要です: len={len(img_path_list)}')
            imgs_list = []
            _uc = True if not (use_rgb_balance or use_rgb_chromatic) else use_rgb_core
            for i in range(6):
                if use_list[i] != 1:
                    continue
                bgr = cv2.imread(img_path_list[i], cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f'無染色入力の読込失敗: {img_path_list[i]}')
                imgs_list.append(planes_from_bgr_modal(bgr, color, use_rgb_balance, use_rgb_chromatic, use_rgb_core=_uc))
            img = np.concatenate(imgs_list, axis=2)
        elif len(use_list)==9:
            if 1 in use_list[0:3]:
                img_list.append(_get_image(img_path_list[0], color, use_list[0:3]))
            if 1 in use_list[3:6]:
                img_list.append(_get_image(img_path_list[1], color, use_list[3:6]))
            if 1 in use_list[6:9]:
                img_list.append(_get_image(img_path_list[2], color, use_list[6:9]))
            img = cv2.merge(img_list)
        elif len(use_list)==18:
            if 1 in use_list[0:3]:
                img_list.append(_get_image(img_path_list[0], 'RGB', use_list[0:3]))
            if 1 in use_list[3:6]:
                img_list.append(_get_image(img_path_list[1], 'RGB', use_list[3:6]))
            if 1 in use_list[6:9]:
                img_list.append(_get_image(img_path_list[2], 'RGB', use_list[6:9]))
            if 1 in use_list[9:12]:
                img_list.append(_get_image(img_path_list[0], 'HSV', use_list[9:12]))
            if 1 in use_list[12:15]:
                img_list.append(_get_image(img_path_list[1], 'HSV', use_list[12:15]))
            if 1 in use_list[15:18]:
                img_list.append(_get_image(img_path_list[2], 'HSV', use_list[15:18]))
            img = cv2.merge(img_list)
        else:
            assert Exception(f'Invalid use_list length: {len(use_list)}')
        img = torchvision.transforms.ToTensor()(img)
        img = torch.reshape(img, (-1, *img.size()))
        return img
    elif blend == 'alpha':
        bf = cv2.imread(img_path_list[0],cv2.IMREAD_COLOR)
        df = cv2.imread(img_path_list[1],cv2.IMREAD_COLOR)
        ph = cv2.imread(img_path_list[2],cv2.IMREAD_COLOR)
        if color == 'RGB':
            bf = cv2.cvtColor(bf, cv2.COLOR_BGR2RGB)
            df = cv2.cvtColor(df, cv2.COLOR_BGR2RGB)
            ph = cv2.cvtColor(ph, cv2.COLOR_BGR2RGB)
        elif color == 'HSV':
            bf = cv2.cvtColor(bf, cv2.COLOR_BGR2HSV)
            df = cv2.cvtColor(df, cv2.COLOR_BGR2HSV)
            ph = cv2.cvtColor(ph, cv2.COLOR_BGR2HSV)
        img = bf * use_list[0] + df * use_list[1] + ph * use_list[2]
        img = img.astype(np.uint8)
        img = torchvision.transforms.ToTensor()(img)
        img = torch.reshape(img, (-1, *img.size()))
        return img

def get_gray_image(img_path):
    img = Image.open(img_path).convert('L')
    img = torchvision.transforms.ToTensor()(img)
    img = torch.reshape(img, (-1, img.size(0), img.size(1), img.size(2)))
    return img
