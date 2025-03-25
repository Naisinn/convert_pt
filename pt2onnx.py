#!/usr/bin/env python3
import torch
import os

def main():
    # 変換対象の.ptファイルのパスを入力
    pt_path = input("変換対象の.ptファイルのパスを入力してください: ").strip()
    
    # 入力されたパスが存在するか確認
    if not os.path.isfile(pt_path):
        print("指定されたファイルが見つかりません。パスを確認してください。")
        return

    # モデルを読み込み（ここではtorch.save(model, path)で保存されたモデルを想定）
    try:
        import models.yolo  # safe_globalsで使用するために必要なモジュール
    except ImportError as e:
        print(f"必要なモジュール 'models.yolo' をインポートできませんでした: {e}")
        return

    try:
        # safe_globalsを利用して、DetectionModelを許可
        with torch.serialization.safe_globals({'models.yolo.DetectionModel': models.yolo.DetectionModel}):
            model = torch.load(pt_path, weights_only=True)
    except Exception as e:
        print(f"モデルの読み込み中にエラーが発生しました: {e}")
        return

    # 推論モードに設定
    model.eval()
    
    # ダミー入力の作成（ここでは入力サイズ(1, 3, 224, 224)を仮定）
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 保存先のONNXファイルパス（.ptの拡張子を.onnxに変更）
    base, _ = os.path.splitext(pt_path)
    onnx_path = base + ".onnx"
    
    try:
        # モデルをONNX形式にエクスポート
        torch.onnx.export(
            model,                    # 変換するモデル
            dummy_input,              # ダミー入力
            onnx_path,                # 出力ファイルのパス
            export_params=True,       # 学習済みパラメータを含む
            opset_version=11,         # ONNXのopsetバージョン
            do_constant_folding=True, # 定数折り畳みの最適化を有効にする
            input_names=['input'],    # 入力ノード名
            output_names=['output']   # 出力ノード名
        )
        print(f"ONNXファイルが保存されました: {onnx_path}")
    except Exception as e:
        print(f"ONNXへの変換中にエラーが発生しました: {e}")

if __name__ == '__main__':
    main()
