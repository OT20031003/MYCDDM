import dataplot
from Diffusion.Train import *
from train import *

# 使用するGPUのIDを指定（ここでは0番目のGPUを使用）
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

def seed_torch(seed=1024):
    """
    実験の再現性を確保するために、各種乱数シードを固定する関数。
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # ハッシュのランダム化を禁止
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # マルチGPUを使用する場合の設定
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class experiment():
    """
    実験全体のハイパーパラメータや条件を定義する静的クラス。
    ここで指定したパラメータの組み合わせでループが回ります。
    """
    test="CDDM"          # 実験モードの指定 ("CDDM", "DnCNN", "GAN", "small" など)
    loss_function = ["MSE"] # 損失関数
    channel_type = ["awgn"] # 通信チャネルの種類 ("awgn", "rayleigh" など)
    dataset = ["DIV2K"]     # 使用するデータセット
    SNRs = [10]             # テスト時のSNR (Signal-to-Noise Ratio)
    train_snr = [13]        # 学習時のSNR
    C_confirm = [36]        # 圧縮後の次元数（C）のリスト
    C_CIFAR = [24,16, 12, 8] # CIFAR10用のC設定候補
    C_DIV2K = [36,24, 12]    # DIV2K用のC設定候補
    #C_CelebA = [48, 24, 16, 8]
    C=[C_DIV2K]             # 実際に使用するCリスト
    CBR_snr=15              # 特定の比較用SNR設定
    large_snr=3             # 高SNR設定（差分学習用など）
    noise_schedule=[1]      # 拡散モデルのノイズスケージュール設定ID
    Tmax=[10]               # 拡散ステップ数などに関連するパラメータ


class config():
    """
    JSCC（合同ソース・チャネル符号化）モデルおよび学習環境の設定クラス。
    データセットごとに異なるパスやモデル構造（Swin Transformerのパラメータなど）を定義します。
    """
    def __init__(self, loss, channel_type, dataset, SNRs, C, encoder_path, decoder_path, re_decoder_path):
        self.loss_function = loss
        self.channel_type = channel_type
        self.database_address = "mongodb://localhost:27017" # 結果保存用DBのアドレス
        self.dataset = dataset
        self.SNRs = SNRs
        self.C = C
        self.seed = 1024
        self.CUDA = True
        self.device = torch.device("cuda:0") # 使用デバイス
        # self.device_ids = [0]  # サーバー環境に合わせて変更
        
        # --- 学習のハイパーパラメータ ---
        self.learning_rate = 0.0001
        self.h_sigma = [0.1, 0.05]
        self.all_SNRs = [20, 15, 10, 5]
        self.n_d_train=4
        
        # --- データセット別の設定分岐 ---
        if self.dataset == "CIFAR10":
            self.test_batch=100
            self.epoch = 200
            self.retrain_epoch=4          # 再学習（Fine-tuning）のエポック数
            self.retrain_save_model_freq=4 # 再学習時の保存頻度
            self.save_model_freq = 200    # 通常学習時の保存頻度
            self.batch_size = 180*4       # バッチサイズ
            self.CDDM_batch=100           # CDDM学習時のバッチサイズ
            self.image_dims = (3, 32, 32) # 画像サイズ (CH, H, W)
            self.train_data_dir = r"/home/wutong/dataset/CIFAR10" # 学習データパス
            self.test_data_dir = r"/home/wutong/dataset/CIFAR10"  # テストデータパス
            # エンコーダ（Swin Transformer）の構造定義
            self.encoder_kwargs = dict(
                img_size=(self.image_dims[1], self.image_dims[2]), patch_size=2, in_chans=3,
                embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8],
                window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=torch.nn.LayerNorm, patch_norm=False,
            )
            # デコーダの構造定義
            self.decoder_kwargs = dict(
                img_size=(self.image_dims[1], self.image_dims[2]),
                embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4],
                window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=torch.nn.LayerNorm, patch_norm=False,
            )

        elif self.dataset == "DIV2K":
            self.test_batch=1
            self.epoch = 600  # 学習エポック数
            self.retrain_epoch=20
            self.retrain_save_model_freq=20
            self.save_model_freq = 600
            self.batch_size = 4  # サーバーのメモリに合わせて調整
            self.CDDM_batch=16
            
            self.image_dims = (3, 256, 256)
            self.train_data_dir = r"/mnt/wutong/datasets/DIV2K/DIV2K_train_HR"
            self.test_data_dir = r"/mnt/wutong/datasets/DIV2K/DIV2K_valid_HR"
            self.encoder_kwargs = dict(
                img_size=(self.image_dims[1], self.image_dims[2]), patch_size=2, in_chans=3,
                embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=torch.nn.LayerNorm, patch_norm=True,
            )
            self.decoder_kwargs = dict(
                img_size=(self.image_dims[1], self.image_dims[2]),
                embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=torch.nn.LayerNorm, patch_norm=True,
            )
        elif self.dataset == "CelebA":
            self.epoch = 1
            self.save_model_freq = 1
            self.batch_size = 50 

            self.image_dims = (3, 128, 128)
            self.train_data_dir = r"D:\dateset\CelebA\Img\trainset"
            self.test_data_dir = r"D:\dateset\CelebA\Img\validset"
            self.encoder_kwargs = dict(
                img_size=(self.image_dims[1], self.image_dims[2]), patch_size=2, in_chans=3,
                embed_dims=[128, 192, 256], depths=[2, 2, 6], num_heads=[4, 6, 8],
                window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=torch.nn.LayerNorm, patch_norm=True,
            )
            self.decoder_kwargs = dict(
                img_size=(self.image_dims[1], self.image_dims[2]),
                embed_dims=[256, 192, 128], depths=[6, 2, 2], num_heads=[8, 6, 4],
                window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=torch.nn.LayerNorm, patch_norm=True,
            )

        # モデルの保存/読み込みパスの設定
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.re_decoder_path = re_decoder_path


class CHDDIM_config():
    """
    拡散モデル（CDDM）専用の設定クラス。
    拡散プロセスや学習に関するパラメータを定義します。
    """
    device_ids = [0]
    epoch = 400            # CDDMの学習エポック数
    save_model_freq = 400  # モデル保存頻度
    T = 1000               # 拡散プロセスの総ステップ数 (Timesteps)
    channel_mult = [1, 2, 2] # U-Netのチャンネル倍率
    attn = [1]             # Attentionブロックを入れる解像度レベル
    num_res_blocks = 2     # ResBlockの数
    dropout = 0.1
    lr = 1e-4              # 学習率
    multiplier = 2.
    snr_max = 1e-4         # スケジューリング用のSNRパラメータ（最大）
    snr_min = 0.02         # スケジューリング用のSNRパラメータ（最小）
    grad_clip = 1          # 勾配クリッピング
    # equ = None
    device = "cuda"
    re_weight=True         # 損失関数の重み付けを行うか


    def __init__(self, C, path, large_snr, noise_schedule, t_max):
        self.C = C
        self.t_max = t_max
        self.noise_schedule = noise_schedule
        self.channel = int(16. * C)  # 潜在表現のチャンネル数を計算（Cに依存）
        self.save_path = path        # CDDMモデルの保存パス
        self.large_snr = large_snr   # 高SNR条件の設定


if __name__ == '__main__':
    # --- 実験ループの開始 ---
    # 定義されたデータセット、損失関数、チャネルタイプなどを総当たりで実行
    for index, dataset in enumerate(experiment.dataset):
        for loss in experiment.loss_function:
            for channel_type in experiment.channel_type:
                # 学習用SNRとテスト用SNRを結合してループ
                SNRs = experiment.train_snr + experiment.SNRs
                # print(SNRs)
                for SNR in SNRs:
                    for noise_schedule in experiment.noise_schedule:
                        for t_max in experiment.Tmax:
                            # チェックポイント（学習済みモデル）のベースディレクトリ
                            basepath = r'/mnt/wutong/CDDMcheckpoints/checkpoints' 
                            
                            # --- パスの生成ロジック ---
                            # 実験条件に応じて、読み込むべきモデルファイル名（.pt）を動的に生成しています
                            if noise_schedule==1:
                                # 標準的なJSCCのエンコーダ・デコーダパス
                                encoder_path = basepath + r'/JSCC/{}/{}/SNRs/encoder_snr{}_channel_{}_C{}.pt'.format(dataset, loss,
                                                                                                                    SNR,
                                                                                                                    channel_type,
                                                                                                                    experiment.C_confirm[index])
                                decoder_path = basepath + r'/JSCC/{}/{}/SNRs/decoder_snr{}_channel_{}_C{}.pt'.format(dataset, loss,
                                                                                                                    SNR,
                                                                                                                    channel_type,
                                                                                                                    experiment.C_confirm[index])
                                # 実験モード（experiment.test）に応じた分岐
                                if experiment.test=="CDDM":
                                    # CDDM用の再学習デコーダ（Re-Decoder）と拡散モデル（CDDM）のパス
                                    re_decoder_path = basepath + r'/JSCC/{}/{}/SNRs/redecoder_snr{}_channel_{}_C{}.pt'.format(dataset,
                                                                                                                            loss,
                                                                                                                            SNR - 3, # SNRを少しずらして学習させる設定か
                                                                                                                            channel_type,
                                                                                                                            experiment.C_confirm[index])
                                    CDDM_path = basepath + r'/CDDM/{}/{}/SNRs/CDDM_snr{}_channel_{}_C{}.pt'.format(dataset, loss, SNR,
                                                                                                                channel_type,
                                                                                                                experiment.C_confirm[index])
                                elif experiment.test=="small":
                                    # 小規模モデル用のパス設定
                                    re_decoder_path = basepath + r'/JSCC/{}/{}/SNRs/redecoder_snr{}_channel_{}_C{}_small.pt'.format(dataset,
                                                                                                                            loss,
                                                                                                                            SNR - 3,
                                                                                                                            channel_type,
                                                                                                                            experiment.C_confirm[index])
                                    CDDM_path = basepath + r'/CDDM/{}/{}/SNRs/CDDM_snr{}_channel_{}_C{}_small.pt'.format(dataset, loss, SNR,
                                                                                                                channel_type,
                                                                                                                experiment.C_confirm[index])
                                elif experiment.test=="lessT":
                                    # ステップ数削減（Less T）モデル用のパス設定
                                    re_decoder_path = basepath + r'/JSCC/{}/{}/SNRs/redecoder_snr{}_channel_{}_C{}_lessT.pt'.format(dataset,
                                                                                                                            loss,
                                                                                                                            SNR - 3,
                                                                                                                            channel_type,
                                                                                                                            experiment.C_confirm[index])
                                    CDDM_path = basepath + r'/CDDM/{}/{}/SNRs/CDDM_snr{}_channel_{}_C{}_lessT.pt'.format(dataset, loss, SNR,
                                                                                                                channel_type,
                                                                                                                experiment.C_confirm[index])
                                elif experiment.test=="re-weight":
                                    # 重み付け変更モデル用のパス設定
                                    re_decoder_path = basepath + r'/JSCC/{}/{}/SNRs/redecoder_snr{}_channel_{}_C{}_noweight.pt'.format(dataset,
                                                                                                                            loss,
                                                                                                                            SNR - 3,
                                                                                                                            channel_type,
                                                                                                                            experiment.C_confirm[index])
                                    CDDM_path = basepath + r'/CDDM/{}/{}/SNRs/CDDM_snr{}_channel_{}_C{}_noreweight.pt'.format(dataset, loss, SNR,
                                                                                                                channel_type,
                                                                                                                experiment.C_confirm[index])
                                elif experiment.test=="Tmax":
                                    # Tmaxを指定したモデル用のパス設定
                                    re_decoder_path = basepath + r'/JSCC/{}/{}/SNRs/redecoder_snr{}_channel_{}_C{}_tmax{}.pt'.format(dataset,
                                                                                                                            loss,
                                                                                                                            SNR - 3,
                                                                                                                            channel_type,
                                                                                                                            experiment.C_confirm[index],t_max)
                                    CDDM_path = basepath + r'/CDDM/{}/{}/SNRs/CDDM_snr{}_channel_{}_C{}.pt'.format(dataset, loss, SNR,
                                                                                                                channel_type,
                                                                                                                experiment.C_confirm[index])
                                elif experiment.test=="DnCNN":
                                    # 比較手法：DnCNN用のパス設定
                                    re_decoder_path = basepath + r'/JSCC/{}/{}/SNRs/redecoder_snr{}_channel_{}_C{}_DnCNN.pt'.format(dataset,
                                                                                                                            loss,
                                                                                                                            SNR - 3,
                                                                                                                            channel_type,
                                                                                                                            experiment.C_confirm[index])
                                    CDDM_path = basepath + r'/CDDM/{}/{}/SNRs/CDDM_snr{}_channel_{}_C{}_DnCNN.pt'.format(dataset, loss, SNR,
                                                                                                                channel_type,
                                                                                                                experiment.C_confirm[index])
                                elif experiment.test=="GAN":
                                    # 比較手法：GAN用のパス設定
                                    re_decoder_path = basepath + r'/JSCC/{}/{}/SNRs/redecoder_snr{}_channel_{}_C{}_GAN.pt'.format(dataset,
                                                                                                                            loss,
                                                                                                                            SNR - 3,
                                                                                                                            channel_type,
                                                                                                                            experiment.C_confirm[index])
                                    CDDM_path = basepath + r'/CDDM/{}/{}/SNRs/CDDM_snr{}_channel_{}_C{}_GAN.pt'.format(dataset, loss, SNR,
                                                                                                                channel_type,
                                                                                                                experiment.C_confirm[index])
                                else:
                                    raise ValueError
                                                                                                                
                            else:
                                # noise_scheduleが1以外の場合のパス設定（通常は1を使用）
                                encoder_path = basepath + r'/JSCC/{}/{}/SNRs/encoder_snr{}_channel_{}_C{}.pt'.format(dataset, loss,
                                                                                                                    SNR,
                                                                                                                    channel_type,
                                                                                                                    experiment.C_confirm[index])
                                decoder_path = basepath + r'/JSCC/{}/{}/SNRs/decoder_snr{}_channel_{}_C{}.pt'.format(dataset, loss,
                                                                                                                    SNR,
                                                                                                                    channel_type,
                                                                                                                    experiment.C_confirm[index])
                                re_decoder_path = basepath + r'/JSCC/{}/{}/SNRs/redecoder_snr{}_channel_{}_C{}_ns{}.pt'.format(dataset,
                                                                                                                        loss,
                                                                                                                        SNR - 3,
                                                                                                                        channel_type,
                                                                                                                        experiment.C_confirm[index],noise_schedule)
                                CDDM_path = basepath + r'/CDDM/{}/{}/SNRs/CDDM_snr{}_channel_{}_C{}_ns{}.pt'.format(dataset, loss, SNR,
                                                                                                            channel_type,
                                                                                                            experiment.C_confirm[index],noise_schedule)
                            
                            # --- 設定オブジェクトのインスタンス化 ---
                            # JSCC（基本通信部）の設定を作成
                            JSCC_config = config(loss=loss, channel_type=channel_type, dataset=dataset, SNRs=SNR,
                                                C=experiment.C_confirm[index], encoder_path=encoder_path,
                                                decoder_path=decoder_path, re_decoder_path=re_decoder_path)
                            # CDDM（拡散モデル部）の設定を作成
                            CDDM_config = CHDDIM_config(C=experiment.C_confirm[index], path=CDDM_path, large_snr=experiment.large_snr, noise_schedule=noise_schedule, t_max=t_max)
                            
                            # シードをリセットして再現性を確保
                            seed_torch()
                            
                            # --- 実行フェーズ ---
                            # 1. まずベースラインとなるJSCC単体の評価を実行
                            # train_JSCC_seqeratly(JSCC_config) # 学習用（コメントアウト中）
                            eval_only_JSCC(JSCC_config) # 評価用
                            
                            # if SNR == max(experiment.SNRs):
                            #     eval_JSCC_SNRs(JSCC_config)
                            # if channel_type == 'rayleigh':
                            #     #eval_only_JSCC_delte_h(JSCC_config)
                            #     pass

                            # 2. 指定されたSNRが学習対象リストに含まれる場合、追加モデル（CDDM等）の学習・評価を実行
                            if SNR in experiment.train_snr:
                                print(experiment.test) # 現在の実験モードを表示
                                
                                if experiment.test=="DnCNN":
                                    # DnCNN（従来のノイズ除去CNN）の学習・評価
                                    seed_torch()
                                    #train_DnCNN(JSCC_config, CDDM_config)
                                    seed_torch()
                                    #train_JSCC_with_DnCNN(JSCC_config, CDDM_config)
                                    seed_torch()
                                    eval_JSCC_with_DnCNN(JSCC_config, CDDM_config)

                                elif experiment.test=="GAN":
                                    # GANベースの手法の学習・評価
                                    seed_torch()
                                    train_GAN(JSCC_config,CDDM_config)
                                    seed_torch()
                                    train_JSCC_with_GAN(JSCC_config,CDDM_config)
                                    seed_torch()
                                    eval_JSCC_with_GAN(JSCC_config,CDDM_config)

                                else:
                                    # デフォルト：CDDM（提案手法）の学習・評価
                                    seed_torch()
                                    #train_CHDDIM(JSCC_config, CDDM_config) # CDDM単体の学習
                                    seed_torch()
                                    #train_JSCC_with_CDDM(JSCC_config, CDDM_config) # 全体のFine-tuning
                                    seed_torch()
                                    eval_JSCC_with_CDDM(JSCC_config, CDDM_config) # 最終評価（ここがメイン）