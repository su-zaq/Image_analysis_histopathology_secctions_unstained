import pathlib
from typing import Optional, Tuple

import torch
from torch.utils import data
import torchvision
from PIL import Image
import cv2
import numpy as np

from rgb_balance import planes_from_bgr_modal
class Dataset(data.Dataset):
    def __init__(
        self,
        folder_path,
        use_list,
        color='RGB',
        blend='concatenate',
        other_channel=False,
        modality_folder_names: Optional[Tuple[str, ...]] = None,
        use_rgb_balance: bool = False,
        use_rgb_chromatic: bool = False,
        use_rgb_core: bool = True,
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
        self.bf_img_paths = []
        self.df_img_paths = []
        self.ph_img_paths = []
        self.y_membrane_img_paths = []
        self.y_nuclear_img_paths = []
        if modality_folder_names is not None:
            self.mod_img_paths = [[] for _ in modality_folder_names]
            for path in folder_path:
                for i, name in enumerate(modality_folder_names):
                    self.mod_img_paths[i] += self._get_file_path(path + '/' + name)
                self.y_membrane_img_paths += self._get_file_path(path + '/y_membrane')
                self.y_nuclear_img_paths += self._get_file_path(path + '/y_nuclear')
            n0 = len(self.mod_img_paths[0]) if self.mod_img_paths else 0
            if n0 == 0:
                raise ValueError('both データセット: 無染色6入力の画像が0枚です。')
            for i in range(len(modality_folder_names)):
                if len(self.mod_img_paths[i]) != n0:
                    raise ValueError(f'無染色6入力: モダリティ間で枚数不一致 slot0={n0} slot{i}={len(self.mod_img_paths[i])}')
        else:
            for path in folder_path:
                self.bf_img_paths += self._get_file_path(path+'/bf')
                self.df_img_paths += self._get_file_path(path+'/df')
                self.ph_img_paths += self._get_file_path(path+'/ph')
                self.y_membrane_img_paths += self._get_file_path(path+'/y_membrane')
                self.y_nuclear_img_paths += self._get_file_path(path+'/y_nuclear')
        self.use_list = use_list
        self.color = color
        self.blend = blend
        self.other_channel = other_channel
        self.use_rgb_balance = use_rgb_balance
        self.use_rgb_chromatic = use_rgb_chromatic
        self.use_rgb_core = True if not (use_rgb_balance or use_rgb_chromatic) else use_rgb_core
        self.to_tensor = torchvision.transforms.ToTensor()

        if modality_folder_names is not None and (blend != 'concatenate' or len(use_list) != 6):
            raise Exception('both 無染色6入力では blend=concatenate かつ len(use_list)==6 が必要です。')
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
        bgr = cv2.imread(paths[index], cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f'画像の読み込みに失敗しました index={index} path={paths[index]}')
        return planes_from_bgr_modal(
            bgr, self.color, self.use_rgb_balance, self.use_rgb_chromatic, use_rgb_core=self.use_rgb_core,
        )

    def __getitem__(self, index):
        if self.blend == 'concatenate':
            img_list = []
            if len(self.use_list)==3:#撮像法のみの検討
                if self.use_list[0]==1:
                    img_list.append(self._get_image(self.bf_img_paths, index, self.color))
                if self.use_list[1]==1:
                    img_list.append(self._get_image(self.df_img_paths, index, self.color))
                if self.use_list[2]==1:
                    img_list.append(self._get_image(self.ph_img_paths, index, self.color))
                x = cv2.merge(img_list)
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
            y_membrane = cv2.imread(self.y_membrane_img_paths[index],cv2.IMREAD_GRAYSCALE)
            y_nuclear = cv2.imread(self.y_nuclear_img_paths[index],cv2.IMREAD_GRAYSCALE)
            if self.other_channel:
                y_other = np.ones_like(y_membrane, dtype=np.float32) - y_membrane.astype(np.float32) - y_nuclear.astype(np.float32)
                y_other = np.where(y_other>0, y_other, 0).astype(np.uint8)
                y = cv2.merge([y_membrane, y_nuclear, y_other])
            else:
                y = cv2.merge([y_membrane, y_nuclear])
            return self.to_tensor(x),self.to_tensor(y)
        elif self.blend == 'alpha':
            bf = cv2.imread(self.bf_img_paths[index],cv2.IMREAD_COLOR)
            df = cv2.imread(self.df_img_paths[index],cv2.IMREAD_COLOR)
            ph = cv2.imread(self.ph_img_paths[index],cv2.IMREAD_COLOR)
            y_membrane = cv2.imread(self.y_membrane_img_paths[index],cv2.IMREAD_GRAYSCALE)
            y_nuclear = cv2.imread(self.y_nuclear_img_paths[index],cv2.IMREAD_GRAYSCALE)
            if self.other_channel:
                y_other - np.ones_like(y_membrane, dtype=np.float32) - y_membrane.astype(np.float32) - y_nuclear.astype(np.float32)
                y_other = np.where(y_other>0, y_other, 0).astype(np.uint8)
                y = cv2.merge([y_membrane, y_nuclear, y_other])
            else:
                y = cv2.merge([y_membrane, y_nuclear])
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
        return len(self.bf_img_paths)

    def _get_file_path(self,path):
        folder_path = pathlib.Path(path)
        img_path = sorted((p for p in folder_path.glob('*') if p.is_file()), key=lambda p: p.name)
        img_path = [str(p) for p in img_path]
        return img_path

def get_dataloader(folder_path, use_list, color='RGB', blend='concatenate', other_channel=False, batch_size = 1, num_workers=0, isShuffle=True, pin_memory=True, modality_folder_names: Optional[Tuple[str, ...]] = None, use_rgb_balance: bool = False, use_rgb_chromatic: bool = False, use_rgb_core: bool = True):
    dataset = Dataset(
        folder_path, use_list, color=color, blend=blend, other_channel=other_channel,
        modality_folder_names=modality_folder_names,
        use_rgb_balance=use_rgb_balance,
        use_rgb_chromatic=use_rgb_chromatic,
        use_rgb_core=use_rgb_core,
    )
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

def get_image(img_path_list, use_list, color='RGB', blend='concatenate', use_rgb_balance: bool = False, use_rgb_chromatic: bool = False, use_rgb_core: bool = True):
    if blend == 'concatenate':
        img_list = []
        if len(use_list)==3:#撮像法のみの検討
            if use_list[0]==1:
                img_list.append(_get_image(img_path_list[0], color))
            if use_list[1]==1:
                img_list.append(_get_image(img_path_list[1], color))
            if use_list[2]==1:
                img_list.append(_get_image(img_path_list[2], color))
            img = cv2.merge(img_list)
        elif len(use_list) == 6:
            if len(img_path_list) < 6:
                raise Exception(f'無染色6入力では img_path_list に6経路が必要です: len={len(img_path_list)}')
            uc = True if not (use_rgb_balance or use_rgb_chromatic) else use_rgb_core
            imgs = []
            for i in range(6):
                if use_list[i] != 1:
                    continue
                bgr = cv2.imread(img_path_list[i], cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f'無染色入力読込失敗 {img_path_list[i]}')
                imgs.append(planes_from_bgr_modal(bgr, color, use_rgb_balance, use_rgb_chromatic, use_rgb_core=uc))
            img = np.concatenate(imgs, axis=2)
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
