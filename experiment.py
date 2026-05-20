'''
薄い標本の細胞膜、細胞核の抽出実験用のプログラム。
通常、画像は最低3分割できるような枚数が必要。
N分割(N>2)にした場合、学習:評価:テスト = N-2:1:1 となる。
2分割の場合、学習:テスト = 1:1 となる。
'''
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False
formatter = Formatter('%(asctime)s : %(levelname)7s - %(message)s')
handler.setFormatter(formatter)

import argparse
import ast
import configparser
import json
import time
import os
import pathlib

from send_info_discord import discord_info

import cv2
import numpy as np
from PIL import Image
import torch
assert torch.cuda.is_available(), 'CUDAが利用できません。'
from torch import optim, nn, autocast
from torch.amp import GradScaler
from torch.nn import BCELoss
import torchvision
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import VitLib
from VitLib import (
    image_processing,
    get_file_paths,
    get_file_stems, 
    file_exists, 
    create_directory,
    modify_line_width,
    make_nuclear_evaluate_images,
)
from VitLib_PyTorch.Loss import DiceLoss, FMeasureLoss, IoULoss, ReverseIoULoss
from VitLib_PyTorch.Network import U_Net, Nested_U_Net

from Dataset import Dataset_experiment_both, Dataset_experiment_single, Dataset_experiment_plus
from rgb_balance import rgb_balance_grayscale, rgb_chromatic_diff_grayscale


def _count_image_files_in_dir(dir_path: str) -> int:
    """VitLib の get_file_paths は「直下のサブフォルダ」だけ返す場合があるため、枚数確認はファイル数を直接数える。"""
    p = pathlib.Path(dir_path).expanduser()
    try:
        p = p.resolve()
    except OSError:
        pass
    if not p.is_dir():
        return 0
    return sum(1 for x in p.iterdir() if x.is_file())


def imwrite_unicode(path: str, img: np.ndarray) -> None:
    """cv2.imwrite の代替。Windows では非 ASCII を含むパスに imwrite が失敗することがある。"""
    p = pathlib.Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower() if p.suffix else '.png'
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f'cv2.imencode に失敗しました: {p}')
    p.write_bytes(buf.tobytes())

# 撮像法名略称
BRIGHT_FIELD = 'bf'
DARK_FIELD = 'df'
PHASE_CONTRAST = 'ph'

_SUBJECT_MEMBRANE_FAMILY = ('membrane', 'membrane_balance')
_SUBJECT_NUCLEAR_FAMILY = ('nuclear', 'nuclear_balance')
_SUBJECT_BALANCE_ONLY = ('membrane_balance', 'nuclear_balance')
_SUBJECT_SINGLE_NET_OUT1 = ('membrane', 'nuclear', 'membrane_balance', 'nuclear_balance')

class Extraction:
    def __init__(
            self,
            experiment_subject:str='membrane',
            use_Network:str='U-Net',
            deep_supervision:bool=False,
            color:str='RGB',
            blend:str='concatenate',
            blend_particle_size:float=0.2,
            gradation:bool=True,
            train_dont_care:bool=False,
            care_rate = 75,
            lower_ratio = 17,
            higher_ratio = 0,
            use_other_channel:bool=False,
            use_softmax:bool=False,
            use_loss:str='DiceLoss',
            start_num:int=0,
            num_epochs:int=40,
            lr:float=5e-05,
            batch_size:int=32,
            use_list_length:int=3,
            img_path:str='./Data/master_exp_data',
            data_augmentation_num:int=500,
            train_size:tuple=(256, 256),
            saturation_mag:tuple=(0.7, 1.3),
            value_mag:tuple=(0.7, 1.3),
            contrast_mag:tuple=(0.7, 1.3),
            radius_train:int=3,
            radius_eval:int=3,
            use_device:list=[i for i in range(torch.cuda.device_count())],
            use_autocast:bool=True,
            autocast_dtype:torch.dtype=torch.float16,
            default_path:str='./result',
            compress_rate:int=1,
            ignore_error:bool=False,
            use_rgb_balance:bool=False,
            use_rgb_chromatic:bool=False,
        ) -> None:
        # 実験パラメータ
        ## 比較対称になる条件
        ### 実験対象(膜, 核)
        #### 膜のみ
        #### 核のみ
        #### 膜と核
        #### 膜マスクを使用して核を抽出
        #### 核マスクを使用して膜を抽出 
        self.experiment_subject = experiment_subject
        assert self.experiment_subject in ['membrane', 'nuclear', 'both', 'nuclear+', 'membrane+', 'membrane_balance', 'nuclear_balance'], f'実験対象が不正です。experiment_subject : {self.experiment_subject}'

        self.use_rgb_balance = use_rgb_balance
        self.use_rgb_chromatic = use_rgb_chromatic

        ### 使用ネットワーク(U-Net, U-Net++)
        self.use_Network = use_Network
        assert self.use_Network in ['U-Net', 'U-Net++'], f'使用ネットワークが不正です。use_Network : {self.use_Network}'

        ### Deep Supervisionの有無
        if self.use_Network == 'U-Net++':
            self.deep_supervision = deep_supervision

        ### 使用色空間(self.use_list_lengthが3または9の場合、RGB, HSV)
        self.color = color
        assert self.color in ['RGB', 'HSV'], f'使用色空間が不正です。color : {self.color}'

        ### 画像の結合方法(concatenate, alpha)
        self.blend = blend
        assert self.blend in ['concatenate', 'alpha'], f'画像の結合方法が不正です。blend : {self.blend}'
        if self.blend == 'alpha':
            #### alphaブレンドの粒度
            self.blend_particle_size = blend_particle_size

        ### 細胞膜の正解画像の膨張時にグラデーションにするか、しないか
        ### マスク学習時もグラデーションにするか，しないか
        if self.experiment_subject in _SUBJECT_MEMBRANE_FAMILY or self.experiment_subject == 'both' or self.experiment_subject == 'nuclear+' or self.experiment_subject == 'membrane+':
            self.gradation = gradation

        ### 細胞核の学習画像にDon't careを含むかどうか
        ### マスク学習時はDon't careを含むかどうか
        if self.experiment_subject in _SUBJECT_NUCLEAR_FAMILY or self.experiment_subject == 'both' or self.experiment_subject == 'nuclear+' or self.experiment_subject == 'membrane+':
            self.train_dont_care = train_dont_care
            self.care_rate = care_rate
            self.lower_ratio = lower_ratio
            self.higher_ratio = higher_ratio

        ### 同時学習時にOther channelを挟むかどうか
        if self.experiment_subject == 'both':
            self.use_other_channel = use_other_channel
            self.use_softmax = use_softmax

        ## 実験全体で固定するパラメータ
        ### 使用するLossの種類(DiceLoss, BCELoss)
        self.use_loss = use_loss
        assert self.use_loss in ['DiceLoss', 'BCELoss', 'FMeasureLoss', 'IoULoss', 'ReverseIoULoss', 'MSELoss'], f'使用Lossが不正です。use_loss : {self.use_loss}'

        ### スタートの番号(途中から再開する場合のみ変更)
        self.start_num = start_num

        ### 合計エポック数
        self.num_epochs = num_epochs

        ### 学習率
        self.lr = lr

        ### バッチサイズ
        self.batch_size = batch_size

        ### 撮像法の利用条件
        # 撮像法単位の場合→3
        # 色空間単位の場合→9
        # RGB, HSV両方使う場合→18
        self.use_list_length = use_list_length
        assert self.use_list_length in [1, 3, 9, 18], f'撮像法の利用条件が不正です。use_list_length : {self.use_list_length}'

        if self.use_rgb_balance or self.use_rgb_chromatic:
            assert self.experiment_subject in ['membrane', 'nuclear'], 'use_rgb_balance / use_rgb_chromatic は membrane または nuclear の単独学習時のみ使用できます。'
            assert self.blend == 'concatenate' and self.use_list_length in (1, 3), (
                'use_rgb_balance / use_rgb_chromatic 時は blend=concatenate かつ use_list_length は 1 または 3 である必要があります。'
                '(1=撮像法を1系統ずつ試す実験、3=bf/df/phのあり/なしをビットで切替)'
            )
        if self.experiment_subject in _SUBJECT_BALANCE_ONLY:
            assert not self.use_rgb_balance and not self.use_rgb_chromatic, (
                'membrane_balance / nuclear_balance では use_rgb_balance / use_rgb_chromatic は False にしてください。'
                '入力は RGB バランス画像のみです。'
            )
            assert self.blend == 'concatenate' and self.use_list_length in (1, 3), (
                'membrane_balance / nuclear_balance 時は blend=concatenate かつ use_list_length は 1 または 3 である必要があります。'
            )

        ### 学習に使用する元の画像(1200px × 1600px)フォルダのパス
        self.img_path = img_path
        """
        **フォルダ構成**
        self.img_path
        ├─pathological_specimen_01(同一標本をまとめるフォルダ, フォルダ名は任意で可)
        │   ├─01(フォルダ名は任意で可)
        │   │  ├─x              (最初から入れておく必要あり→[bf.png, df.png, he.png])
        │   │  └─y_membrane    (最初から入れておく必要あり→[ans_thin.png], 実験開始時に条件に応じて作成→[ans.png])
        │   ├─02
        │   │  ├─x              (最初から入れておく必要あり→[bf.png, df.png, he.png])
        │   │  └─y_membrane    (最初から入れておく必要あり→[ans_thin.png], 実験開始時に条件に応じて作成→[ans.png])
        │   同一標本内の画像の枚数分続く
        ├─pathological_specimen_02
        │   ├─01
        |   ├─02
        |   同一標本内の画像の枚数分続く
        標本数分続く
        """

        ### 1組のデータ辺りの拡張枚数
        self.data_augmentation_num = data_augmentation_num

        ### ネットワーク学習時の画像サイズ
        self.train_size = train_size

        ### 拡張時の画像変化倍率
        #### 彩度の変更倍率
        self.saturation_mag = saturation_mag
        #### 明度の変更倍率
        self.value_mag = value_mag
        #### コントラストの変更倍率
        self.contrast_mag = contrast_mag

        ### 細胞膜の膨張度合い
        #### 学習時
        self.radius_train = radius_train
        #### 評価時(このプログラムでは使用しない)
        self.radius_eval = radius_eval

        ### GPU
        self.use_device = use_device
        self.gpu_n = len(self.use_device)

        ### 学習にautocastを使用するかどうか
        self.use_autocast = use_autocast

        ### 学習時にAutocastを使用する場合のdtype
        self.autocast_dtype = autocast_dtype
        assert self.autocast_dtype in [torch.float16, torch.bfloat16, torch.float32], f'autocast_dtypeが不正です。autocast_dtype : {self.autocast_dtype}'

        ### 記録用のフォルダのパス
        self.default_path = self.set_path(default_path)

        ### 保存時の圧縮倍率(数字を大きくすると圧縮率(一纏めにされる画素数)が上がる)
        self.compress_rate = compress_rate

        ### エラーを無視するための設定
        self.ignore_error = ignore_error

        # 以上が実験で設定するパラメータ
        ####################################################################################################################
        # 実験時に使用するパスの作成、設定
        ## 実験時に学習用データを展開するフォルダ
        self.train_data_folder = self.set_path(self.default_path + '/train_data/')
        ## ログ記録用フォルダ
        self.log_folder = self.set_path(self.default_path + '/log/')
        if not file_exists(self.default_path + '/log/exp.log'):
            file_handler = FileHandler(self.default_path + '/log/exp.log', encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info('logファイルの作成完了')
        elif self.start_num == 0:
            raise Exception(self.default_path + '/log/exp.logは存在します。')
        ## 推論結果画像の保存先
        if self.experiment_subject in _SUBJECT_MEMBRANE_FAMILY or self.experiment_subject == 'membrane+':
            self.save_image_path = self.set_path(self.default_path + f'/eval_data_membrane/')
        elif self.experiment_subject in _SUBJECT_NUCLEAR_FAMILY or self.experiment_subject == 'nuclear+':
            self.save_image_path = self.set_path(self.default_path + f'/eval_data_nuclear/')
        elif self.experiment_subject == 'both':
            self.save_membrane_image_path = self.set_path(self.default_path + '/eval_data_membrane/')
            self.save_nuclear_image_path = self.set_path(self.default_path + '/eval_data_nuclear/')
            
        ####################################################################################################################
        # modelの保存
        self.model_folder = self.set_path(self.default_path + '/model/')
        ####################################################################################################################
        # 外部フォルダ推論（epochごと）
        # 入力：この直下の画像を全部推論（ファイル名は固定しない）
        self.external_input_root = './test/x'
        # 出力：epochごとに保存（exp/roop/test/epoch で分かるように）
        self.external_output_root = self.set_path(self.default_path + '/external_pred/')
        # BGR->RGB をするか（学習がRGB前提なら True 推奨）
        self.external_bgr2rgb = True
        # 学習時に x.max()<=1 なので、推論入力も 0..1 に合わせる
        self.external_div255 = True
        ####################################################################################################################
        # 学習パラメータの記録  
        self.data_param_path = self.default_path + '/log/data_parm.json'
        if self.start_num == 0:
            self.save_json(self.data_param_path, vars(self))
            self.parm_log()
        else:
            self.check_variable()

        ####################################################################################################################
        # 学習画像の作成
        self.pathological_specimen_folder_paths = get_file_paths(self.img_path)
        #>> pathological_specimen_folder_paths = ['Data/master_exp_data/pathological_specimen_01', 'Data/master_exp_data/pathological_specimen_02', 'Data/master_exp_data/pathological_specimen_03', 'Data/master_exp_data/pathological_specimen_04']
        assert len(self.pathological_specimen_folder_paths) > 1, f'画像は2分割以上である必要があります。pathological_specimen_folder_paths : {self.pathological_specimen_folder_paths}'
        if self.start_num == 0:
            logger.info('学習時に利用する画像の作成開始')
            pathological_specimen_folder_stems = get_file_stems(self.img_path)
            #>> pathological_specimen_folder_stems = ['pathological_specimen_01', 'pathological_specimen_02', 'pathological_specimen_03', 'pathological_specimen_04']

            for folder_path, folder_stem in zip(self.pathological_specimen_folder_paths, pathological_specimen_folder_stems):
                if self.experiment_subject in _SUBJECT_MEMBRANE_FAMILY:
                    #細線化の正解から学習用の正解を作成する
                    logger.info(folder_path+' の細線化の正解画像から学習用の正解を作成開始')
                    self.make_ans_img_membrane(folder_path)
                elif self.experiment_subject in _SUBJECT_NUCLEAR_FAMILY:
                    #細胞核のDon't care画像を作成する。
                    logger.info(folder_path+' の核の正解画像から学習用の正解を作成開始')
                    self.make_ans_img_nuclear(folder_path)
                elif self.experiment_subject == 'both':
                    logger.info(folder_path+' の細線化の正解画像から学習用の正解を作成開始')
                    self.make_ans_img_membrane(folder_path)
                    logger.info(folder_path+' の核の正解画像から学習用の正解を作成開始')
                    self.make_ans_img_nuclear(folder_path)
                elif self.experiment_subject == 'membrane+' or self.experiment_subject == 'nuclear+':
                    logger.info(folder_path+' の細線化の正解画像から学習用の正解を作成開始')
                    self.make_ans_img_membrane(folder_path)
                    logger.info(folder_path+' の核の正解画像から学習用の正解を作成開始')
                    self.make_ans_img_nuclear(folder_path)
                else:
                    raise Exception(f'実験対象が不正です。experiment_subject : {self.experiment_subject}')
                #データ拡張
                logger.info(folder_path+' のデータ拡張画像を作成開始')
                self.proc_img(folder_path, self.train_data_folder + folder_stem)
        ####################################################################################################################
        # パターン数の計算
        # 2のuse_list_length乗から全て使用しない1パターンを引いた値
        if self.blend == 'concatenate':
            if self.use_list_length == 1:
                self.use_lists = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                self.img_pattern = len(self.use_lists)
            if self.use_list_length == 3:
                self.img_pattern = 2 ** self.use_list_length -1
                self.use_lists = [self.get_use_list(n+1, self.use_list_length) for n in range(self.img_pattern)]
            #if self.use_list_length==3:
            #>> self.use_lists = [[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
            #>> self.use_lists[n] =[明視野の使用有無(0:使用しない, 1:使用), 暗視野の使用有無, 位相差の使用有無]
            #if self.use_list_length==9:
            #>> self.use_lists[n] =[明R(H), 明G(S), 明B(V), 暗R(H), 暗G(S), 暗B(V), 位R(H), 位G(S), 位B(V)]
            #if self.use_list_length==18:
            #>> self.use_lists[n] =[明R, 明G, 明B, 暗R, 暗G, 暗B, 位R, 位G, 位B, 明H, 明S, 明V, 暗H, 暗S, 暗V, 位H, 位S, 位V]
        elif self.blend == 'alpha':
            self.use_lists = []
            self.rate_dict = {}
            exp_num = 1
            n = int(1/self.blend_particle_size) + 1
            for bf in range(n):
                for df in range(n):
                    bf_rate = bf * self.blend_particle_size
                    df_rate = df * self.blend_particle_size
                    ph_rate = 1 - bf_rate - df_rate
                    if 0<=ph_rate<=1:
                        self.use_lists.append([bf_rate, df_rate, ph_rate])
                        self.rate_dict[f'exp{exp_num:04d}'] = [bf_rate, df_rate, ph_rate]
                        exp_num += 1
            self.save_json(f'{self.log_folder}rate_dict.json', self.rate_dict)
            self.img_pattern = len(self.use_lists)

        self.data_set_folder_path_list = get_file_paths(self.train_data_folder)
        #>> self.data_set_folder_path_list = ['{{self.default_path}}/train_data/pathological_specimen_01', '{{self.default_path}}/train_data/pathological_specimen_02', '{{self.default_path}}/train_data/pathological_specimen_03', '{{self.default_path}}/train_data/pathological_specimen_04']
        self.roop_num = len(self.data_set_folder_path_list)

        ####################################################################################################################
        # 実験開始
        logger.info('実験開始')
        self.exp_roop()
        logger.info('実験終了')

    def set_path(self, path:str) -> str:
        """パスを設定し、フォルダを作成する関数

        Args:
            path (str): パス

        Returns:
            str: パス
        """
        create_directory(path)
        return path

    def save_json(self, path:str, data:dict) -> None:
        """データをjsonファイルに保存する関数

        Args:
            path (str): 保存先のパス
            data (dict): 保存するデータ
        """
        with open(path, 'w', encoding='UTF-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False, default=self.custom_encoder)

    def custom_encoder(self, obj) -> str:
        if not isinstance(obj, str):
            return str(obj)

    def parm_log(self) -> None:
        """パラメータの記録を行う関数"""
        logger.info(f'実験対象 : {self.experiment_subject}')
        logger.info(f'使用ネットワーク : {self.use_Network}')
        logger.info(f'使用色空間 : {self.color}')
        if self.experiment_subject in _SUBJECT_MEMBRANE_FAMILY or self.experiment_subject == 'both':
            logger.info(f'細胞膜の正解画像の膨張時にグラデーションにするか、しないか : {self.gradation}')
        if self.experiment_subject in _SUBJECT_NUCLEAR_FAMILY or self.experiment_subject == 'both':
            logger.info(f'細胞核の学習画像にDon\'t careを含むかどうか : {self.train_dont_care}')
            logger.info(f'細胞核のDon\'t care画像の割合 : {self.care_rate}')
            logger.info(f'細胞核のDon\'t care画像の下限割合 : {self.lower_ratio}')
            logger.info(f'細胞核のDon\'t care画像の上限割合 : {self.higher_ratio}')
        if self.experiment_subject == 'both':
            logger.info(f'同時学習時にOther channelを挟むかどうか : {self.use_other_channel}')
        logger.info(f'使用Loss : {self.use_loss}')
        logger.info(f'スタートの番号 : {self.start_num}')
        logger.info(f'合計エポック数 : {self.num_epochs}')
        logger.info(f'学習率 : {self.lr}')
        logger.info(f'バッチサイズ : {self.batch_size}')
        logger.info(f'撮像法の利用条件 : {self.use_list_length}')
        logger.info(f'学習に使用する元の画像(1200px × 1600px)フォルダのパス : {self.img_path}')
        logger.info(f'1組のデータ辺りの拡張枚数 : {self.data_augmentation_num}')
        logger.info(f'ネットワーク学習時の画像サイズ : {self.train_size}')
        logger.info(f'彩度の変更倍率 : {self.saturation_mag}')
        logger.info(f'明度の変更倍率 : {self.value_mag}')
        logger.info(f'コントラストの変更倍率 : {self.contrast_mag}')
        logger.info(f'細胞膜の膨張度合い(学習時) : {self.radius_train}')
        logger.info(f'細胞膜の膨張度合い(評価時) : {self.radius_eval}')
        logger.info(f'GPU数 : {self.gpu_n}')
        logger.info(f'学習にautocastを使用するかどうか : {self.use_autocast}')
        logger.info(f'学習時にAutocastを使用する場合のdtype : {self.autocast_dtype}')
        logger.info(f'記録用のフォルダのパス : {self.default_path}')
        logger.info(f'保存時の圧縮倍率 : {self.compress_rate}')
        logger.info(f'エラーを無視するための設定 : {self.ignore_error}')
        if self.experiment_subject in ['membrane', 'nuclear']:
            logger.info(f'RGBバランスチャンネル追加 : {self.use_rgb_balance}, RGB色差チャンネル追加 : {self.use_rgb_chromatic}')
        if self.experiment_subject in _SUBJECT_BALANCE_ONLY:
            logger.info('入力チャンネル : RGB バランス画像のみ（撮像法ごと 3ch、use_list は従来どおり）')
        logger.info(f'学習時のデータ展開先フォルダ : {self.train_data_folder}')
        logger.info(f'ログ記録用フォルダ : {self.log_folder}')
        if self.experiment_subject in _SUBJECT_SINGLE_NET_OUT1:
            logger.info(f'推論結果画像の保存先 : {self.save_image_path}')
        elif self.experiment_subject == 'both':
            logger.info(f'細胞膜の推論結果画像の保存先 : {self.save_membrane_image_path}')
            logger.info(f'細胞核の推論結果画像の保存先 : {self.save_nuclear_image_path}')

    def load_json(self, path:str) -> dict:
        """jsonファイルからデータを読み込む関数

        Args:
            path (str): パス

        Returns:
            dict: 読み込んだデータ
        """
        with open(path, 'r', encoding='UTF-8') as f:
            return json.load(f)

    def check_variable(self) -> None:
        """変数の一致を確認する関数"""
        past = self.load_json(self.data_param_path)
        current = vars(self)

        if past.keys()==current.keys():#変数の種類が一致する場合
            for k in past.keys():
                if past[k]!=current[k]:
                    logger.warning(f'変数{k}の値が一致していません。新 : {current[k]}, 旧 : {past[k]}')
        else:#変数の種類が一致しない場合
            no_match_list = []
            for k in past.keys():
                if k in current.keys():
                    if past[k]!=current[k]:
                        no_match_list.append(k)
                        logger.warning(f'変数{k}の値が一致していません。新 : {current[k]}, 旧 : {past[k]}')
                else:
                    logger.warning(f'変数{k}がありません。')
            
            add_variable = False
            for k in current.keys():
                if k not in past.keys():
                    logger.warning(f'変数{k}が追加されています。')
                    add_variable = True
            
            if add_variable:
                self.save_json(self.default_path + '/log/data_parm.json', vars(self))
                self.parm_log()

    def make_ans_single_img_membrane(self, in_path:str, out_path:str) -> None:
        """細線化の正解画像から学習用の正解画像を作成する関数

        Args:
            in_path (str): 読み込む画像のパス
            out_path (str): 画像の保存先パス
        """
        img_thin = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        img_thin = np.array(img_thin/255, dtype=np.uint8)
        result = np.zeros_like(img_thin) + img_thin
        if self.gradation:
            for i in range(1, self.radius_train+1):
                result += modify_line_width(img_thin, radius=i)
            result = np.around(result / (self.radius_train+1), decimals=1)
        else:
            result = modify_line_width(img_thin, radius=self.radius_train)
        imwrite_unicode(out_path, result*255)

    def make_ans_img_membrane(self, img_folder_path:str) -> None:
        """細線化の正解画像から学習用の正解画像を作成する関数
        
        Args:
            img_folder_path (str): 画像フォルダのパス
        """
        for img_path in get_file_paths(img_folder_path):
            if self.gradation:
                self.make_ans_single_img_membrane(f'{img_path}/y_membrane/ans_thin.png', f'{img_path}/y_membrane/ans.png')
            else:
                self.make_ans_single_img_membrane(f'{img_path}/y_membrane/ans_thin.png', f'{img_path}/y_membrane/ans_nograd.png')

    def make_ans_single_img_nuclear(self, in_ans_path:str, in_bf_path:str, out_eval_img_path:str, out_red_img:str, out_green_img:str) -> None:
        """核の正解画像と明視野画像からDon't care画像を作成する関数

        Args:
            in_ans_path (str): 正解画像のパス
            in_bf_path (str): 明視野画像のパス
            out_eval_img_path (str): 評価画像の保存先パス
            out_red_img (str): 赤チャンネルの保存先パス
            out_green_img (str): 緑チャンネルの保存先パス
        """
        ans_img = cv2.imread(in_ans_path, cv2.IMREAD_GRAYSCALE)
        bf_img = cv2.imread(in_bf_path, cv2.IMREAD_COLOR)
        result = make_nuclear_evaluate_images(ans_img, bf_img, self.care_rate, self.lower_ratio, self.higher_ratio)
        imwrite_unicode(out_eval_img_path, result['eval_img'])
        imwrite_unicode(out_red_img, result['red_img'])
        imwrite_unicode(out_green_img, result['green_img'])

    def make_ans_img_nuclear(self, img_folder_path:str) -> None:
        """核の正解画像と明視野画像からDon't care画像を作成する関数
        
        Args:
            img_folder_path (str): 画像フォルダのパス
        """
        for img_path in get_file_paths(img_folder_path):
            self.make_ans_single_img_nuclear(f'{img_path}/y_nuclear/ans.png', f'{img_path}/x/{BRIGHT_FIELD}.png', f'{img_path}/y_nuclear/eval.png', f'{img_path}/y_nuclear/red.png', f'{img_path}/y_nuclear/green.png')

    def _rgb_extra_channel_multiplier(self) -> int:
        """膜・核単独時: 1 + (バランス3ch) + (色差3ch) のブロック数。"""
        if self.experiment_subject not in ('membrane', 'nuclear'):
            return 1
        m = 1
        if self.use_rgb_balance:
            m += 1
        if self.use_rgb_chromatic:
            m += 1
        return m

    def proc_img(self, img_folder_path:str, save_folder_path:str) -> None:
        """画像の拡張を行う関数

        Args:
            img_folder_path (str): 画像フォルダのパス
            save_folder_path (str): 保存先のパス
        """

        # 保存先フォルダの作成
        if self.experiment_subject not in _SUBJECT_BALANCE_ONLY:
            create_directory(f'{save_folder_path}/{BRIGHT_FIELD}')
            create_directory(f'{save_folder_path}/{DARK_FIELD}')
            create_directory(f'{save_folder_path}/{PHASE_CONTRAST}')
        if self.experiment_subject in ('membrane', 'nuclear'):
            create_directory(f'{save_folder_path}/y')
            if self.use_rgb_balance:
                create_directory(f'{save_folder_path}/{BRIGHT_FIELD}_bal')
                create_directory(f'{save_folder_path}/{DARK_FIELD}_bal')
                create_directory(f'{save_folder_path}/{PHASE_CONTRAST}_bal')
            if self.use_rgb_chromatic:
                create_directory(f'{save_folder_path}/{BRIGHT_FIELD}_cdiff')
                create_directory(f'{save_folder_path}/{DARK_FIELD}_cdiff')
                create_directory(f'{save_folder_path}/{PHASE_CONTRAST}_cdiff')
        elif self.experiment_subject in _SUBJECT_BALANCE_ONLY:
            create_directory(f'{save_folder_path}/y')
            create_directory(f'{save_folder_path}/{BRIGHT_FIELD}_bal')
            create_directory(f'{save_folder_path}/{DARK_FIELD}_bal')
            create_directory(f'{save_folder_path}/{PHASE_CONTRAST}_bal')
        elif self.experiment_subject == 'membrane+' or self.experiment_subject == 'nuclear+':
            create_directory(f'{save_folder_path}/y_membrane')
            create_directory(f'{save_folder_path}/y_nuclear')
        elif self.experiment_subject == 'both':
            create_directory(f'{save_folder_path}/y_membrane')
            create_directory(f'{save_folder_path}/y_nuclear')
        else:
            raise Exception(f'実験対象が不正です。experiment_subject : {self.experiment_subject}')

        # 画像が既に存在する場合は実験中断の可能性があるため、エラーを出力
        if self.experiment_subject in _SUBJECT_BALANCE_ONLY:
            _exist_check_dir = f'{save_folder_path}/{BRIGHT_FIELD}_bal'
        else:
            _exist_check_dir = f'{save_folder_path}/{BRIGHT_FIELD}'
        img_num = _count_image_files_in_dir(_exist_check_dir)
        if not self.ignore_error and img_num > 0:
            logger.error(f'{_exist_check_dir} に画像が存在します。')
            raise Exception(f'{_exist_check_dir} に画像が存在します。')

        img_subfolders = get_file_paths(img_folder_path)
        if len(img_subfolders) == 0:
            raise RuntimeError(
                f'proc_img: マスタ側に画像組フォルダがありません: get_file_paths({img_folder_path!r}) が空です。'
                f'README のとおり、各標本フォルダ直下に x/, y_membrane/ 等がある「画像組」サブフォルダを1つ以上置いてください。'
                f' save_folder_path={save_folder_path!r}'
            )

        for img_path in img_subfolders:
            # 画像の読み込み
            logger.info(f'{img_path} の画像を作成中')
            bf_img = cv2.imread(f'{img_path}/x/{BRIGHT_FIELD}.png', cv2.IMREAD_COLOR)
            df_img = cv2.imread(f'{img_path}/x/{DARK_FIELD}.png', cv2.IMREAD_COLOR)
            he_img = cv2.imread(f'{img_path}/x/{PHASE_CONTRAST}.png', cv2.IMREAD_COLOR)
            
            # 画像の読み込みチェック
            bf_path = f'{img_path}/x/{BRIGHT_FIELD}.png'
            df_path = f'{img_path}/x/{DARK_FIELD}.png'
            he_path = f'{img_path}/x/{PHASE_CONTRAST}.png'
            
            if bf_img is None:
                raise Exception(f'画像の読み込みに失敗しました: {bf_path}')
            if df_img is None:
                raise Exception(f'画像の読み込みに失敗しました: {df_path}')
            if he_img is None:
                raise Exception(f'画像の読み込みに失敗しました: {he_path}')
            
            if self.experiment_subject in _SUBJECT_MEMBRANE_FAMILY:
                if self.gradation:
                    ans_img = cv2.imread(f'{img_path}/y_membrane/ans.png', cv2.IMREAD_GRAYSCALE)
                else:
                    ans_img = cv2.imread(f'{img_path}/y_membrane/ans_nograd.png', cv2.IMREAD_GRAYSCALE)
                if ans_img is None:
                    p = f'{img_path}/y_membrane/ans.png' if self.gradation else f'{img_path}/y_membrane/ans_nograd.png'
                    raise Exception(f'正解画像を読み込めませんでした（make_ans_img 等の事前準備を確認）: {p}')
            elif self.experiment_subject in _SUBJECT_NUCLEAR_FAMILY:
                if self.train_dont_care:
                    ans_img = cv2.imread(f'{img_path}/y_nuclear/ans.png', cv2.IMREAD_GRAYSCALE)
                else:
                    ans_img = cv2.imread(f'{img_path}/y_nuclear/green.png', cv2.IMREAD_GRAYSCALE)
                if ans_img is None:
                    p = f'{img_path}/y_nuclear/ans.png' if self.train_dont_care else f'{img_path}/y_nuclear/green.png'
                    raise Exception(f'正解画像を読み込めませんでした: {p}')
            elif self.experiment_subject == 'membrane+' or self.experiment_subject == 'nuclear+':
                if self.gradation:
                    ans_mem_img = cv2.imread(f'{img_path}/y_membrane/ans.png', cv2.IMREAD_GRAYSCALE)
                else:
                    ans_mem_img = cv2.imread(f'{img_path}/y_membrane/ans_nograd.png', cv2.IMREAD_GRAYSCALE)
                if self.train_dont_care:
                    ans_nuc_img = cv2.imread(f'{img_path}/y_nuclear/ans.png', cv2.IMREAD_GRAYSCALE)
                else:
                    ans_nuc_img = cv2.imread(f'{img_path}/y_nuclear/green.png', cv2.IMREAD_GRAYSCALE)
            elif self.experiment_subject == 'both':
                if self.gradation:
                    ans_mem_img = cv2.imread(f'{img_path}/y_membrane/ans.png', cv2.IMREAD_GRAYSCALE)
                else:
                    ans_mem_img = cv2.imread(f'{img_path}/y_membrane/ans_nograd.png', cv2.IMREAD_GRAYSCALE)
                if self.train_dont_care:
                    ans_nuc_img = cv2.imread(f'{img_path}/y_nuclear/ans.png', cv2.IMREAD_GRAYSCALE)
                else:
                    ans_nuc_img = cv2.imread(f'{img_path}/y_nuclear/green.png', cv2.IMREAD_GRAYSCALE)
            else:
                raise Exception(f'実験対象が不正です。experiment_subject : {self.experiment_subject}')

            for i in range(self.data_augmentation_num):
                if i % (self.data_augmentation_num//10) == 0:
                    logger.info(f'{i}/{self.data_augmentation_num} の画像を作成中...')
                if self.experiment_subject in _SUBJECT_SINGLE_NET_OUT1:
                    img_list = [bf_img, df_img, he_img, ans_img]
                elif self.experiment_subject == 'membrane+' or self.experiment_subject == 'nuclear+':
                    img_list = [bf_img, df_img, he_img, ans_mem_img, ans_nuc_img]
                elif self.experiment_subject == 'both':
                    img_list = [bf_img, df_img, he_img, ans_mem_img, ans_nuc_img]
                else:
                    raise Exception(f'実験対象が不正です。experiment_subject : {self.experiment_subject}')
                img_list = image_processing.random_cut_image(img_list, self.train_size)
                img_list = image_processing.random_flip_image(img_list)
                img_list = image_processing.random_rotate_image(img_list)
                img_list = image_processing.random_value(img_list, self.value_mag)
                img_list = image_processing.random_saturation(img_list, self.saturation_mag)
                img_list = image_processing.random_contrast(img_list, self.contrast_mag)
                if self.experiment_subject not in _SUBJECT_BALANCE_ONLY:
                    imwrite_unicode(f'{save_folder_path}/{BRIGHT_FIELD}/{img_num:05d}.png', img_list[0])
                    imwrite_unicode(f'{save_folder_path}/{DARK_FIELD}/{img_num:05d}.png', img_list[1])
                    imwrite_unicode(f'{save_folder_path}/{PHASE_CONTRAST}/{img_num:05d}.png', img_list[2])
                gen_balance = self.experiment_subject in _SUBJECT_BALANCE_ONLY or (
                    self.experiment_subject in ('membrane', 'nuclear') and self.use_rgb_balance
                )
                gen_cdiff = self.experiment_subject in ('membrane', 'nuclear') and self.use_rgb_chromatic
                if gen_balance or gen_cdiff:
                    for raw, sub in (
                        (img_list[0], BRIGHT_FIELD),
                        (img_list[1], DARK_FIELD),
                        (img_list[2], PHASE_CONTRAST),
                    ):
                        if gen_balance:
                            bb, gg, rr = rgb_balance_grayscale(raw)
                            bal_bgr = cv2.merge([bb, gg, rr])
                            imwrite_unicode(f'{save_folder_path}/{sub}_bal/{img_num:05d}.png', bal_bgr)
                        if gen_cdiff:
                            rb, gb, rg = rgb_chromatic_diff_grayscale(raw)
                            cd_bgr = cv2.merge([rb, gb, rg])
                            imwrite_unicode(f'{save_folder_path}/{sub}_cdiff/{img_num:05d}.png', cd_bgr)
                if self.experiment_subject in ('membrane', 'nuclear') or self.experiment_subject in _SUBJECT_BALANCE_ONLY:
                    imwrite_unicode(f'{save_folder_path}/y/{img_num:05d}.png', img_list[3])
                elif self.experiment_subject == 'membrane+' or self.experiment_subject == 'nuclear+':
                    imwrite_unicode(f'{save_folder_path}/y_membrane/{img_num:05d}.png', img_list[3])
                    imwrite_unicode(f'{save_folder_path}/y_nuclear/{img_num:05d}.png', img_list[4])
                elif self.experiment_subject == 'both':
                    imwrite_unicode(f'{save_folder_path}/y_membrane/{img_num:05d}.png', img_list[3])
                    imwrite_unicode(f'{save_folder_path}/y_nuclear/{img_num:05d}.png', img_list[4])
                else:
                    raise Exception(f'実験対象が不正です。experiment_subject : {self.experiment_subject}')
                img_num += 1
            logger.info(f'{img_path} の画像の作成完了')

        if self.experiment_subject in _SUBJECT_BALANCE_ONLY or self.experiment_subject in ('membrane', 'nuclear'):
            ny = _count_image_files_in_dir(f'{save_folder_path}/y')
            if ny == 0:
                raise RuntimeError(
                    f'proc_img: 学習用画像が1枚も出力されませんでした: {save_folder_path}/y/ 。'
                    f'マスタ {img_folder_path} 直下の画像組フォルダと正解パスを確認してください。'
                )
        elif self.experiment_subject in ('membrane+', 'nuclear+'):
            for sub, name in (('y_membrane', '膜'), ('y_nuclear', '核')):
                n = _count_image_files_in_dir(f'{save_folder_path}/{sub}')
                if n == 0:
                    raise RuntimeError(f'proc_img: {save_folder_path}/{sub}/ が空です（{name}）。')
        elif self.experiment_subject == 'both':
            for sub, name in (('y_membrane', '膜'), ('y_nuclear', '核')):
                n = _count_image_files_in_dir(f'{save_folder_path}/{sub}')
                if n == 0:
                    raise RuntimeError(f'proc_img: {save_folder_path}/{sub}/ が空です（{name}）。')

    def get_use_list(self, n:int, length:int) -> list:
        """use_listを生成する関数

        Args:
            n (int): 生成するリストの番号
            length (int): 生成するリストの長さ

        Returns:
            list: 生成したリスト
        """
        return [n >> i & 1 for i in range(length)]

    def exp_roop(self) -> None:
        """実験全体のループ部分"""
        for i in range(self.start_num, len(self.use_lists)):
            self.use_list = self.use_lists[i]
            self.exp_num = i + 1
            for self.j in range(self.roop_num):
                self.train_path_list = self.data_set_folder_path_list.copy()
                self.test_path = self.train_path_list.pop(self.j)#テストに使用するファイルパス
                self.test_num = self.data_set_folder_path_list.index(self.test_path)#テストに使用するインデックス
                if self.roop_num > 2:
                    self.val_path = self.train_path_list.pop(self.j-1)#評価に使用するファイルパス
                    self.val_num = self.data_set_folder_path_list.index(self.val_path)#評価に使用するインデックス
                logger.info(f'experiment: {self.exp_num}/{self.img_pattern} - roop_num: {self.j + 1} / {self.roop_num} - all_roop_num: {(self.exp_num - 1) * self.roop_num + self.j + 1} / {self.roop_num * self.img_pattern}')
                self.train_roop()

    def train_roop(self) -> None:
        """撮像法の組み合わせ単位のループ部分"""
        if self.gpu_n > 1:
            self.device = torch.device(f'cuda')
        else:
            self.device = torch.device(f'cuda:{self.use_device[0]}')

        # Define model
        if self.blend == 'alpha':
            in_channels = 3
        elif self.use_list_length == 1:
            in_channels = 3 * self._rgb_extra_channel_multiplier()
        elif self.use_list_length == 3:
            if self.experiment_subject == 'membrane+' or self.experiment_subject == 'nuclear+':
                in_channels = sum(self.use_list) * 3 + 3
            else:
                in_channels = sum(self.use_list) * 3 * self._rgb_extra_channel_multiplier()
        else:
            in_channels = sum(self.use_list)

        self.current_in_channels = in_channels

        if self.use_Network == 'U-Net':
            if self.experiment_subject in _SUBJECT_SINGLE_NET_OUT1:
                self.model = U_Net(in_channels, 1, bilinear=False).to(self.device, non_blocking=True)
            elif self.experiment_subject == 'membrane+' or self.experiment_subject == 'nuclear+':
                self.model = U_Net(in_channels, 1, bilinear=False).to(self.device, non_blocking=True)
            elif self.experiment_subject == 'both':
                if self.use_other_channel:
                    self.model = U_Net(in_channels, 3, bilinear=False, softmax=self.use_softmax).to(self.device, non_blocking=True)
                else:
                    self.model = U_Net(in_channels, 2, bilinear=False, softmax=self.use_softmax).to(self.device, non_blocking=True)
            else:
                raise Exception(f'実験対象が不正です。experiment_subject : {self.experiment_subject}')
        elif self.use_Network == 'U-Net++':
            if self.experiment_subject in _SUBJECT_SINGLE_NET_OUT1:
                self.model = Nested_U_Net(in_channels, 1, deepsupervision=self.deep_supervision).to(self.device, non_blocking=True)
            elif self.experiment_subject == 'membrane+' or self.experiment_subject == 'nuclear+':
                self.model = Nested_U_Net(in_channels, 1, deepsupervision=self.deep_supervision).to(self.device, non_blocking=True)
            elif self.experiment_subject == 'both':
                if self.use_other_channel:
                    self.model = Nested_U_Net(in_channels, 3, softmax=self.use_softmax, deepsupervision=self.deep_supervision).to(self.device, non_blocking=True)
                else:
                    self.model = Nested_U_Net(in_channels, 2, softmax=self.use_softmax, deepsupervision=self.deep_supervision).to(self.device, non_blocking=True)
            else:
                raise Exception(f'実験対象が不正です。experiment_subject : {self.experiment_subject}')
        else:
            raise Exception(f'使用ネットワークが不正です。use_Network : {self.use_Network}')
        
        # if Usable GPU is more than 1, use DataParallel
        if self.gpu_n > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.use_device)

        # Optimizer functions
        if self.use_loss == 'DiceLoss':
            self.criterion = DiceLoss()
        elif self.use_loss == 'BCELoss':
            self.criterion = BCELoss()
        elif self.use_loss == 'FMeasureLoss':
            self.criterion = FMeasureLoss
        elif self.use_loss == 'IoULoss':
            self.criterion = IoULoss()
        elif self.use_loss == 'ReverseIoULoss':
            self.criterion = ReverseIoULoss()
        elif self.use_loss == 'MSELoss':
            self.criterion = nn.MSELoss()
        else:
            raise Exception(f'使用Lossが不正です。use_loss : {self.use_loss}')
        

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        if self.use_autocast:
            self.scaler = GradScaler()

        if len(self.train_path_list) == 0:
            raise RuntimeError(
                '学習に使う train_data サブフォルダが0個です（テスト・検証の除外後）。'
                f' data_set_folder_path_list の数={len(self.data_set_folder_path_list)}, roop_num={self.roop_num} を確認してください。'
            )

        # Data loader
        if self.experiment_subject in _SUBJECT_SINGLE_NET_OUT1:
            self.dataloader = Dataset_experiment_single.get_dataloader(
                self.train_path_list, self.use_list, self.color, self.blend,
                batch_size=self.batch_size, num_workers=2, isShuffle=True, pin_memory=True,
                use_rgb_balance=self.use_rgb_balance,
                use_rgb_chromatic=self.use_rgb_chromatic,
                use_balance_input_only=(self.experiment_subject in _SUBJECT_BALANCE_ONLY),
            )
        elif self.experiment_subject == 'membrane+' or self.experiment_subject == 'nuclear+':
            self.dataloader = Dataset_experiment_plus.get_dataloader(self.train_path_list, self.use_list, self.experiment_subject, self.color, self.blend, batch_size=self.batch_size, num_workers=2, isShuffle=True, pin_memory=True)
        elif self.experiment_subject == 'both':
            self.dataloader = Dataset_experiment_both.get_dataloader(self.train_path_list, self.use_list, self.color, self.blend, self.use_other_channel, batch_size=self.batch_size, num_workers=2, isShuffle=True, pin_memory=True)

        # Training
        for epoch in range(self.num_epochs):
            logger.info(f'experiment: {self.exp_num}/{self.img_pattern} - roop_num: {self.j + 1} / {self.roop_num} - all_roop_num: {(self.exp_num - 1) * self.roop_num + self.j + 1} / {self.roop_num * self.img_pattern} - epoch: {epoch + 1}/{self.num_epochs}')
            self.train()
            ################################################################
            # save_model
            if (epoch + 1) == 20:
                self.save_model(epoch + 1)
            
            self.external_infer(epoch + 1)
            ################################################################
            if self.roop_num > 2:
                # Validation
                img_path_list = self.pathological_specimen_folder_paths[self.val_num]
                if self.experiment_subject in _SUBJECT_SINGLE_NET_OUT1:
                    save_path = self.set_path(f'{self.save_image_path}val/exp{self.exp_num:04d}/val{self.val_num+1:02d}/epoch{epoch+1:02d}/')
                    self.save_image(img_path_list, save_path)
                elif self.experiment_subject == 'membrane+' or self.experiment_subject == 'nuclear+':
                    save_path = self.set_path(f'{self.save_image_path}val/exp{self.exp_num:04d}/val{self.val_num+1:02d}/epoch{epoch+1:02d}/')
                    self.save_image(img_path_list, save_path)
                elif self.experiment_subject == 'both':
                    save_path = self.set_path(f'{self.save_membrane_image_path}val/exp{self.exp_num:04d}/val{self.val_num+1:02d}/epoch{epoch+1:02d}/')
                    save_nuclear_path = self.set_path(f'{self.save_nuclear_image_path}val/exp{self.exp_num:04d}/val{self.val_num+1:02d}/epoch{epoch+1:02d}/')
                    self.save_image(img_path_list, save_path, save_nuclear_path)
            
            # Test
            img_path_list = self.pathological_specimen_folder_paths[self.test_num]
            if self.experiment_subject in _SUBJECT_SINGLE_NET_OUT1:
                save_path = self.set_path(f'{self.save_image_path}test/exp{self.exp_num:04d}/test{self.val_num+1:02d}/epoch{epoch+1:02d}/')
                self.save_image(img_path_list, save_path)
            elif self.experiment_subject == 'membrane+' or self.experiment_subject == 'nuclear+':
                save_path = self.set_path(f'{self.save_image_path}test/exp{self.exp_num:04d}/test{self.val_num+1:02d}/epoch{epoch+1:02d}/')
                self.save_image(img_path_list, save_path)
            elif self.experiment_subject == 'both':
                save_path = self.set_path(f'{self.save_membrane_image_path}test/exp{self.exp_num:04d}/test{self.val_num+1:02d}/epoch{epoch+1:02d}/')
                save_nuclear_path = self.set_path(f'{self.save_nuclear_image_path}test/exp{self.exp_num:04d}/test{self.val_num+1:02d}/epoch{epoch+1:02d}/')
                self.save_image(img_path_list, save_path, save_nuclear_path)

    def image_compression_save(self, pred:torch.Tensor, path:str, index:int=0, divide:int=2, channel:int=0) -> None:
        """推論結果を圧縮して画像保存する関数
        
        Args:
            pred (torch.Tensor): 推論結果
            path (str): 保存先のパス
            index (int, optional): 保存する画像のインデックス(推論バッチサイズが1の場合は指定しなくてよい). Defaults to 0.
            divide (int, optional): 保存する画像の分割数. Defaults to 2.
        """
        #torch.Tensor -> numpy.ndarray(cv2)
        img_torch = pred[index][channel]
        img_cv2 =  np.array(img_torch.cpu().detach().numpy().copy()*255, dtype=np.uint8)

        #要素を圧縮
        img_cv2 = np.where(img_cv2%divide == 0, img_cv2, img_cv2 - img_cv2%divide)

        #numpy.ndarray(cv2) -> PIL.Image.Image
        img_PIL = Image.fromarray(img_cv2)

        #保存
        img_PIL.save(path)

    def save_image(self, img_path_list:str, save_path:str, save_nuclear_path:str=None) -> None:
        """画像を保存する関数
        
        Args:
            img_path_list (str): 推論に使用する画像のPath
            save_path (str): 保存先のパス(self.experiment_subjectが'membrane'または'nuclear'の場合), 細胞膜画像の保存先のパス(self.experiment_subjectが'both'の場合)
            save_nuclear_path (str, optional): 細胞核画像の保存先のパス(self.experiment_subjectが'both'の場合). Defaults to None.
        """
        img_folder_path_list = get_file_paths(img_path_list)
        for img_path in tqdm(img_folder_path_list):
            img_num = img_path.split('/')[-1]
            img_path_list = []
            for img_name in [BRIGHT_FIELD, DARK_FIELD, PHASE_CONTRAST]:
                img_path_list.append(f'{img_path}/x/{img_name}.png')

            self.model.eval()
            if self.experiment_subject in _SUBJECT_SINGLE_NET_OUT1 or self.experiment_subject == 'both':  
                img = Dataset_experiment_single.get_image(
                    img_path_list, self.use_list, self.color, self.blend,
                    use_rgb_balance=self.use_rgb_balance,
                    use_rgb_chromatic=self.use_rgb_chromatic,
                    use_balance_input_only=(self.experiment_subject in _SUBJECT_BALANCE_ONLY),
                )
                img = img.to(self.device)
            elif self.experiment_subject == 'nuclear+':
                for path in img_path_list:
                    base_path = path.split('/x/')[0]
                    ans_path = f'{base_path}/y_membrane/ans_nograd.png'
                    img_path_list.append(ans_path)
                    break
                img = Dataset_experiment_plus.get_image(img_path_list, self.use_list, self.color,self.blend)
                img = img.to(self.device)
            elif self.experiment_subject == 'membrane+':
                for path in img_path_list:
                    base_path = path.split('/x/')[0]
                    ans_path = f'{base_path}/y_nuclear/ans.png'
                    img_path_list.append(ans_path)
                    break
                img = Dataset_experiment_plus.get_image(img_path_list, self.use_list, self.color,self.blend)
                img = img.to(self.device)
            else:
                raise Exception(f'実験対象が不正です。experiment_subject : {self.experiment_subject}')

            with torch.no_grad():
                #Pillowでの画像保存
                #to_pil_image(self.model(img)[0]).save(save_path+str(img_num)+'.png')

                #torchvisionでの画像保存
                #torchvision.utils.save_image(self.model(img)[0], save_path+str(img_num)+'.png')

                #cv2での画像保存
                #pred = self.model(img)[0].permute(1,2,0)
                #pred = np.array(pred.cpu().detach().numpy().copy())*255
                #cv2.imwrite(save_path+str(img_num)+'.png',pred)

                #numpyでの保存(その1)
                #pred = self.model(img)[0].permute(1,2,0)
                #pred = np.array(pred.cpu().detach().numpy().copy()*255, dtype=np.uint8)
                #np.save(save_path+str(img_num)+'.npy', pred[:, :, 0])

                #numpyでの保存(その2)
                #pred = self.model(img)[0].permute(1,2,0)
                #pred = np.array(pred.cpu().detach().numpy().copy()*255, dtype=np.uint8)
                #np.savez_compressed(save_path+str(img_num)+'.npz', img = pred[:, :, 0])

                pred = self.model(img)
                if isinstance(pred, list):
                    pred = pred[-1]
                if self.experiment_subject in _SUBJECT_SINGLE_NET_OUT1:
                    self.image_compression_save(pred, f'{save_path}{img_num}.png', divide=self.compress_rate)
                elif self.experiment_subject == 'membrane+' or self.experiment_subject == 'nuclear+':
                    self.image_compression_save(pred, f'{save_path}{img_num}.png', divide=self.compress_rate)
                elif self.experiment_subject == 'both':
                    self.image_compression_save(pred, f'{save_path}{img_num}.png', divide=self.compress_rate, channel=0)
                    self.image_compression_save(pred, f'{save_nuclear_path}{img_num}.png', divide=self.compress_rate, channel=1)
                del pred
                torch.cuda.empty_cache()
            
    def train(self) -> None:
        """学習用関数(Tensorflowでいうfit関数的なもの)"""
        self.model.train()
        size = len(self.dataloader.dataset)
        bar_format = '{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
        train_roop = tqdm(self.dataloader, bar_format=bar_format)
        average_loss = 0
        current = 0
        for batch, (x, y) in enumerate(train_roop):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            assert x.max() <= 1.0, f'xの最大値が1.0を超えています。x.max : {x.max}'

            if self.use_autocast:
                self.optimizer.zero_grad(set_to_none=True)

                with autocast(device_type='cuda', dtype=self.autocast_dtype):
                    pred = self.model(x)
                    if isinstance(pred, list):
                        loss = 0
                        for p in pred:
                            loss += self.criterion(p, y)
                    else:
                        loss = self.criterion(pred, y)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 4.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.zero_grad(set_to_none=True)

                pred = self.model(x)
                if isinstance(pred, list):
                    loss = 0
                    for p in pred:
                        loss += self.criterion(p, y)
                else:
                    loss = self.criterion(pred, y)

                loss.backward()
                self.optimizer.step()

            average_loss += loss.item()
            current += x.size(0)
            train_roop.set_postfix_str(f'loss: {average_loss/current:>7f}  [{current:>5d}/{size:>5d}]')
    
    def save_model(self, epoch:int) -> None:
        """モデルと学習情報を保存する関数"""
        # DataParallel対応
        model_to_save = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

        save_obj = {
            "epoch": epoch,
            "exp_num": self.exp_num,
            "use_list": self.use_list,
            "roop_j": self.j,
            "state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "use_Network": self.use_Network,
            "experiment_subject": self.experiment_subject,
            "color": self.color,
            "blend": self.blend,
            "use_list_length": self.use_list_length,
            "in_channels": getattr(model_to_save, "in_channels", None),
            "use_rgb_balance": self.use_rgb_balance,
            "use_rgb_chromatic": self.use_rgb_chromatic,
            "use_balance_input_only": self.experiment_subject in _SUBJECT_BALANCE_ONLY,
        }

        # ファイル名：どの実験・どの分割・何epochか分かるように
        path = f"{self.model_folder}exp{self.exp_num:04d}_test{self.test_num+1:02d}_epoch{epoch:02d}.pth"
        torch.save(save_obj, path)
        logger.info(f"Model saved: {path}")
        
    def load_external_image(self, path: str) -> torch.Tensor:
        """外部フォルダ画像を読み込んでモデル入力Tensorを作る（撮像法1つ固定）"""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise Exception(f'画像の読み込みに失敗しました: {path}')

        # 2D -> 3D
        if img.ndim == 2:
            img = img[:, :, None]

        # チャンネル調整：モデルのin_channelsに合わせる
        # U_Net / Nested_U_Net の in_channels を直接持ってないことがあるので、
        # 推論時は「現在のモデルが受け付ける入力チャンネル数」を self.use_list から推定した in_channels と同じにする
        # train_roop()で計算した in_channels を保持しておくのが確実なので、無ければここで再計算する
        if not hasattr(self, "current_in_channels"):
            # train_roop()のin_channels計算と同じ
            if self.blend == 'alpha':
                in_channels = 3
            elif self.use_list_length == 1:
                in_channels = 3 * self._rgb_extra_channel_multiplier()
            elif self.use_list_length == 3:
                if self.experiment_subject == 'membrane+' or self.experiment_subject == 'nuclear+':
                    in_channels = sum(self.use_list) * 3 + 3
                else:
                    in_channels = sum(self.use_list) * 3 * self._rgb_extra_channel_multiplier()
            else:
                in_channels = sum(self.use_list)
            self.current_in_channels = in_channels

        in_channels = self.current_in_channels

        if (
            self.experiment_subject in _SUBJECT_BALANCE_ONLY
            and self.use_list_length in (1, 3)
            and sum(self.use_list) == 1
            and img.shape[2] >= 3
        ):
            bgr_u8 = img[:, :, :3].astype(np.uint8)
            b0, g0, r0 = rgb_balance_grayscale(bgr_u8)
            bal_bgr = cv2.merge([b0, g0, r0])
            if self.color == 'RGB':
                part = cv2.cvtColor(bal_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            else:
                part = cv2.cvtColor(bal_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            if self.external_div255:
                part /= 255.0
            x = torch.from_numpy(part).permute(2, 0, 1)
            if x.shape[0] < in_channels:
                pad = torch.zeros((in_channels - x.shape[0], x.shape[1], x.shape[2]), dtype=x.dtype)
                x = torch.cat([x, pad], dim=0)
            elif x.shape[0] > in_channels:
                x = x[:in_channels]
            return x

        # 膜・核単独 + RGB 拡張チャンネル + 撮像法1成分 use_list のときは 1 枚の BGR から複数ブロックを構成
        if (
            (self.use_rgb_balance or self.use_rgb_chromatic)
            and self.experiment_subject in ('membrane', 'nuclear')
            and self.use_list_length in (1, 3)
            and sum(self.use_list) == 1
            and img.shape[2] >= 3
        ):
            bgr_u8 = img[:, :, :3].astype(np.uint8)
            if self.color == 'RGB':
                part1 = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2RGB).astype(np.float32)
            else:
                part1 = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
            if self.external_div255:
                part1 /= 255.0
            parts = [part1]
            if self.use_rgb_balance:
                b0, g0, r0 = rgb_balance_grayscale(bgr_u8)
                bal_bgr = cv2.merge([b0, g0, r0])
                if self.color == 'RGB':
                    part_b = cv2.cvtColor(bal_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
                else:
                    part_b = cv2.cvtColor(bal_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
                if self.external_div255:
                    part_b /= 255.0
                parts.append(part_b)
            if self.use_rgb_chromatic:
                rb, gb, rg = rgb_chromatic_diff_grayscale(bgr_u8)
                cd_bgr = cv2.merge([rb, gb, rg])
                if self.color == 'RGB':
                    part_c = cv2.cvtColor(cd_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
                else:
                    part_c = cv2.cvtColor(cd_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
                if self.external_div255:
                    part_c /= 255.0
                parts.append(part_c)
            img_f = np.concatenate(parts, axis=2)
            x = torch.from_numpy(img_f).permute(2, 0, 1)
            if x.shape[0] < in_channels:
                pad = torch.zeros((in_channels - x.shape[0], x.shape[1], x.shape[2]), dtype=x.dtype)
                x = torch.cat([x, pad], dim=0)
            elif x.shape[0] > in_channels:
                x = x[:in_channels]
            return x

        # BGR->RGB（cv2はBGR）
        if img.shape[2] >= 3 and in_channels >= 3:
            img = img[:, :, :3]
            if self.external_bgr2rgb:
                img = img[:, :, ::-1]
        else:
            img = img[:, :, :min(img.shape[2], in_channels)]

        # float化
        img = img.astype(np.float32)

        # 学習時と合わせる（x.max()<=1 なので）
        if self.external_div255:
            img /= 255.0

        # HWC -> CHW
        x = torch.from_numpy(img).permute(2, 0, 1)

        # チャンネルが足りない場合は0でパディング
        if x.shape[0] < in_channels:
            pad = torch.zeros((in_channels - x.shape[0], x.shape[1], x.shape[2]), dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)

        # 多い場合は切る
        if x.shape[0] > in_channels:
            x = x[:in_channels]

        return x

    def external_infer(self, epoch: int) -> None:
        """epochごとに外部フォルダ（self.external_input_root）直下の画像を全推論して保存"""
        if not file_exists(self.external_input_root):
            logger.warning(f'external_input_root が存在しません: {self.external_input_root}')
            return

        # 出力先（epochごと）
        out_dir = self.set_path(
            f'{self.external_output_root}'
            f'exp{self.exp_num:04d}/test{self.test_num+1:02d}/epoch{epoch:02d}/'
        )

        # 直下の画像だけ対象
        exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp')
        files = [
            f for f in os.listdir(self.external_input_root)
            if os.path.isfile(os.path.join(self.external_input_root, f)) and f.lower().endswith(exts)
        ]
        if len(files) == 0:
            logger.warning(f'external_input_root に画像がありません: {self.external_input_root}')
            return

        self.model.eval()
        for fname in tqdm(files, desc=f'external_infer epoch{epoch:02d}'):
            in_path = os.path.join(self.external_input_root, fname)

            x = self.load_external_image(in_path).unsqueeze(0).to(self.device, non_blocking=True)

            with torch.no_grad():
                pred = self.model(x)
                if isinstance(pred, list):
                    pred = pred[-1]

                base = os.path.splitext(fname)[0]

                if self.experiment_subject == 'both':
                    self.image_compression_save(pred, f'{out_dir}{base}_mem.png', divide=self.compress_rate, channel=0)
                    self.image_compression_save(pred, f'{out_dir}{base}_nuc.png', divide=self.compress_rate, channel=1)
                else:
                    self.image_compression_save(pred, f'{out_dir}{base}.png', divide=self.compress_rate, channel=0)

            del x, pred
            torch.cuda.empty_cache()



if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--config', '-c', type=str, default=None, help='config file path')
    args = arg.parse_args()

    if args.config is not None:
        perser = configparser.ConfigParser()
        perser.read(args.config, encoding='utf-8')
        EXPERIMENT_PATH = perser['EXPERIMENT_PATH']
        EXPERIMENT_PARAM = perser['EXPERIMENT_PARAM']
    else:
        EXPERIMENT_PATH = {}
        EXPERIMENT_PARAM = {}

    def _cfg_bool(key: str, default: bool = False) -> bool:
        if hasattr(EXPERIMENT_PARAM, 'getboolean'):
            return EXPERIMENT_PARAM.getboolean(key, fallback=default)
        return default

    experiment_subject = EXPERIMENT_PARAM.get('experiment_subject', 'membrane')
    use_Network = EXPERIMENT_PARAM.get('use_Network', 'U-Net')
    deep_supervision = _cfg_bool('deep_supervision', False)
    color = EXPERIMENT_PARAM.get('color', 'RGB')
    blend = EXPERIMENT_PARAM.get('blend', 'concatenate')
    gradation = _cfg_bool('gradation', False)
    train_dont_care = _cfg_bool('train_dont_care', False)
    care_rate = float(EXPERIMENT_PARAM.get('care_rate', 75))
    lower_ratio = float(EXPERIMENT_PARAM.get('lower_ratio', 17))
    higher_ratio = float(EXPERIMENT_PARAM.get('higher_ratio', 0))
    use_other_channel = _cfg_bool('use_other_channel', False)
    use_softmax = _cfg_bool('use_softmax', False)
    start_num = int(EXPERIMENT_PARAM.get('start_num', 0))
    num_epochs = int(EXPERIMENT_PARAM.get('num_epochs', 40))
    lr = float(EXPERIMENT_PARAM.get('lr', 5e-4))
    batch_size = int(EXPERIMENT_PARAM.get('batch_size', 32))
    use_list_length = int(EXPERIMENT_PARAM.get('use_list_length', 3))
    img_path = EXPERIMENT_PATH.get('img_path', './Data/master_exp_data')
    data_augmentation_num = int(EXPERIMENT_PARAM.get('data_augmentation_num', 500))
    train_size = ast.literal_eval(EXPERIMENT_PARAM.get('train_size', '(256, 256)'))
    saturation_mag = ast.literal_eval(EXPERIMENT_PARAM.get('saturation_mag', '(0.7, 1.3)'))
    value_mag = ast.literal_eval(EXPERIMENT_PARAM.get('value_mag', '(0.7, 1.3)'))
    contrast_mag = ast.literal_eval(EXPERIMENT_PARAM.get('contrast_mag', '(0.7, 1.3)'))
    radius_train = int(EXPERIMENT_PARAM.get('radius_train', 3))
    radius_eval = int(EXPERIMENT_PARAM.get('radius_eval', 3))
    use_device = ast.literal_eval(EXPERIMENT_PARAM.get('use_device', '[0]'))
    use_autocast = _cfg_bool('use_autocast', False)
    autocast_dtype_text = EXPERIMENT_PARAM.get('autocast_dtype', 'float16')
    try:
        autocast_dtype = getattr(torch, autocast_dtype_text)
    except AttributeError:
        autocast_dtype = torch.bfloat16
    default_path = EXPERIMENT_PATH.get('default_path', './result')
    compress_rate = int(EXPERIMENT_PARAM.get('compress_rate', 1))
    ignore_error = _cfg_bool('ignore_error', False)
    use_rgb_balance = _cfg_bool('use_rgb_balance', False)
    use_rgb_chromatic = _cfg_bool('use_rgb_chromatic', False)

    Extraction(
        experiment_subject=experiment_subject,
        use_Network=use_Network,
        color=color,
        blend=blend,
        gradation=gradation,
        train_dont_care=train_dont_care,
        care_rate=care_rate,
        lower_ratio=lower_ratio,
        higher_ratio=higher_ratio,
        start_num=start_num,
        num_epochs=num_epochs,
        lr=lr,
        batch_size=batch_size,
        use_list_length=use_list_length,
        img_path=img_path,
        data_augmentation_num=data_augmentation_num,
        train_size=train_size,
        saturation_mag=saturation_mag,
        value_mag=value_mag,
        contrast_mag=contrast_mag,
        radius_train=radius_train,
        radius_eval=radius_eval,
        use_device=use_device,
        use_autocast=use_autocast,
        autocast_dtype=autocast_dtype,
        default_path=default_path,
        compress_rate=compress_rate,
        ignore_error=ignore_error,
        use_rgb_balance=use_rgb_balance,
        use_rgb_chromatic=use_rgb_chromatic,
    )
    
    discord_info(105)

