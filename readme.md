# RealtimeDeepPredictionCV2

## Get Started

### Requirements

- python
- pytorch
- torchvision
- opencv

## Usage

### Run Sample

動画に対して実行する場合．

```bash
python3 main.py --filepath <input_movie_path> --checkpoint <model_checkpoint_path>
```

接続されたカメラをストリーミングして利用する場合．

```bash
python3 main.py --checkpoint <model_checkpoint_path>   --use_camera
```

### Argument

```bash
python3 main.py --filepath <input_movie_path> --checkpoint <model_checkpoint_path> --use_camera --use_cpu
```

| Argument | type | status | default | discription |
| --- | --- | --- | --- | --- |
| filepath | string | Optional | movie/WIN_20221122_15_59_43_Pro.mp4 | 読み込む動画ファイルのパスを指定する．use_cameraがFalseの場合に有効． |
| checkpoint | string | Optional | model/best.pth | 推論に使用するモデルの重みファイルのパスを指定する．保存時のデバイスに注意． |
| use_camera | bool | Optional | False | 引数に加えることでTrueになり，接続されたカメラを用いたストリーミング推論を行う． |
| use_cpu | bool | Optional | False | 引数に加えることでTrueになり，GPUを認識している場合でもCPUを用いた推論を行う． |
