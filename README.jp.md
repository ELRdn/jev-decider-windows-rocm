# Jev / Decider を Windows ROCm で省メモリ運用

[English](README.md) · [構成](docs/architecture.md) · [検証結果](docs/validation.md)

**Decider-2B v10を維持したまま、待機時のVRAMを解放する**ためのJev Gateway連携です。大きな線形層をINT8で保持し、BF16で計算します。元のチェックポイントは変更しません。

RX 9070 XTで、変更前の常駐量は**約6.18GiB**。変更後の稼働時の採取値は**約3.11～3.70GiB**、待機後にプロセスを再作成すると**0MiB**になりました。稼働時の数値は採取時点の値で、最大消費量の保証ではありません。

**代わりに、完全解放後の初回GPU準備には約25.8秒かかりました。** 準備中や混雑時は通常のCodexへ委譲します。待機時のVRAM削減を優先する構成であり、Codex全体の高速化やトークン削減は未検証です。

## 動作

1. Windowsサインイン時にGatewayとローカルサービスを起動。モデルはRAMへ読み込みます。
2. 判定要求が来たときだけGPUへ移して推論します。
3. GPU処理終了から10秒使われなければRAMへ退避します。
4. 30秒使われなければプロセスの再作成を要求します。監視間隔を含め約30～40秒で旧プロセスを終了し、Windows側に残った確保領域も返します。
5. 新しいサービスはRAMで待機します。GPUを一度も使っていないサービスは繰り返し再作成しません。

GPU推論は1本の専用ワーカーで処理します。混雑は503、過大入力は413を返し、GatewayからCodexへ委譲します。モデルの確率の書き換えや、Jevによる直接ツール実行は行いません。

## 前提

- Windowsと、対象GPUで動作確認済みのROCm/PyTorch環境。実測したGPUはRX 9070 XT / gfx1201です。
- Python 3.12、[Mapika/decider](https://github.com/Mapika/decider)、Transformers、FLA、互換性のあるTriton。
- Hugging Faceにキャッシュ済みの `Mapika/decider-2b` **v10**。サービスはオフラインで読み込み、バージョンを確認します。
- Node.js、Jev Gateway。検証版は `npm install -g jev-gateway@0.4.1`。
- 通常の手順でサインイン済みのCodex。

GPUドライバーやtorch/Tritonを一括インストールするリポジトリではありません。既存の動作する環境へ、必要ならHTTP依存だけ追加します。以下のパスは例です。

```powershell
& 'D:\AI\decider\.venv\Scripts\python.exe' -m pip install -r requirements-service.txt
```

## 導入

更新時は既存サービスを停止してから実行してください。`jev-local stop` はGatewayを残すため、Gateway設定を変更するときは `jev-codex --stop` も実行します。

まず変更予定を確認します。

```powershell
.\scripts\Install.ps1 `
  -PythonPath 'D:\AI\decider\.venv\Scripts\python.exe' `
  -DeciderDirectory 'D:\AI\decider' `
  -RegisterLogonTask -WhatIf
```

実際に導入し、自動起動を登録して起動する場合:

```powershell
.\scripts\Install.ps1 `
  -PythonPath 'D:\AI\decider\.venv\Scripts\python.exe' `
  -DeciderDirectory 'D:\AI\decider' `
  -RegisterLogonTask -StartNow
```

`%USERPROFILE%\.jev-gateway\local` にサービスと個別設定を配置し、[.env.example](.env.example) のローカル接続設定をGatewayへ反映します。置換するファイルはバックアップします。既存の空でないTypeSafeキーや無関係な環境設定は保持します。サンプルのキーはローカル用のダミー値です。

`-RegisterLogonTask` は現在のユーザーのサインイン用タスク **Jev Local Decider** を、管理者権限なしで登録します。`-StartNow` はその場で起動します。両方を省略すればファイル配置と設定のみです。特殊な配置には `-StateDirectory`、`-NodePath`、`-GatewayLauncher` を指定できます。

ファイル生成・バックアップ・環境設定のマージ・パスの引用は隔離したテストで確認しています。別のPCへのドライバーからの新規導入まで検証したものではありません。

### Codexの接続

既存の `%USERPROFILE%\.codex\config.toml` の他の設定を保持し、次をマージします。導入スクリプトはCodexの設定や認証情報を編集しません。

```toml
model_provider = "jev-gateway"

[model_providers.jev-gateway]
name = "Jev Gateway"
base_url = "http://127.0.0.1:8790/v1"
wire_api = "responses"
requires_openai_auth = true
```

上流は `https://chatgpt.com/backend-api/codex` です。Gatewayがすでに動いていた場合は `.env` の変更後に再起動し、必要に応じてCodex側の設定も再読み込みします。このproviderをすでに使用していれば重複追加は不要です。

## 操作

```powershell
$jev = Join-Path $env:USERPROFILE '.jev-gateway\local\jev-local.ps1'
& $jev status
& $jev dashboard
& $jev stop
& $jev start
& $jev restart
```

`ready: true`、`phase: idle`、`gpu_resident: false` は正常なRAM待機です。完全解放後の次のGPU要求は初回準備から始まります。`stop` は次回サインインにも引き継がれ、`start` で解除します。停止中もGatewayによる通常のCodex応答は継続します。

`installation.json` の既定値は `precision: int8`、`idle_seconds: 10`、`deep_idle_seconds: 30`。`precision: bf16` に戻すこともできますが、稼働中のVRAMは増えます。完全解放の待機時間はRAM退避の待機時間より長くしてください。

## 検証と制約

実機ではAPI推論、Gatewayの読み書き判定、同時要求、過大入力、RAMからの再開、自動起動、プロセスの再作成、Windows GPUカウンターを確認しました。22ケース・37問の小規模比較では選択結果と確認対象の確信度区分が一致しましたが、確率は完全には一致しません。一般的な精度維持の証明ではありません。

```powershell
& 'D:\AI\decider\.venv\Scripts\python.exe' -m pip install -r requirements-test.txt
& 'D:\AI\decider\.venv\Scripts\python.exe' -m unittest discover -s tests -v
.\tests\Test-Installer.ps1
& 'D:\AI\decider\.venv\Scripts\python.exe' scripts\probe_service.py
```

CIはWindows上のCPUテストです。ROCm互換性やVRAMは検証しません。Deciderがない場合、Decider固有のパディングテストはスキップします。`probe_service.py` は判定だけを呼び、選択されたツールを実行しません。

公式FP8の演算は測定環境で失敗したため採用していません。`scripts/fp8_probe.py` で別途確認できます。GGUFバックエンドや0.8Bへの切り替えは含みません。詳しくは[測定結果](docs/validation.md)を参照してください。

## ライセンス

Apache-2.0。[LICENSE](LICENSE) と [NOTICE](NOTICE) を参照してください。モデル重み・依存ライブラリはそれぞれのライセンスに従います。TypeSafe/Jev、Mapika、AMDの公式配布ではありません。認証情報、個別PCの設定、会話ログ、モデル重みは収録していません。
