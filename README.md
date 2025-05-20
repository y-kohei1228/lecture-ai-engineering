# Backlog Wiki RAG System

このプロジェクトは、Backlog上のWikiコンテンツを取得し、それをRetrieval Augmented Generation (RAG)システムとして利用するためのツールです。

## 概要

BacklogのWikiコンテンツをOpenAIのエンベディングモデルを使用してベクトル化し、Chromaベクトルデータベースに保存します。
保存された情報に対して、自然言語で質問を投げかけることで、関連する情報を検索・生成することができます。

## 特徴

- Backlog APIを使用してWikiコンテンツを取得
- テキストをチャンク（部分）に分割して効率的な検索を実現
- OpenAIのエンベディングモデルによるテキストのベクトル化
- Chroma DBによるベクトル検索
- 質問応答システムによる情報検索

## 必要条件

- Python 3.8以上
- Backlog APIキー
- OpenAI APIキー

## インストール方法

1. リポジトリをクローンするか、ファイルをダウンロード
2. 依存ライブラリをインストール:

```bash
pip install -r requirements.txt
```

## 使い方

1. 環境変数を設定:

```bash
# Windows
set BACKLOG_SPACE=あなたのスペース名
set BACKLOG_API_KEY=あなたのBacklog APIキー
set OPENAI_API_KEY=あなたのOpenAI APIキー

# macOS/Linux
export BACKLOG_SPACE=あなたのスペース名
export BACKLOG_API_KEY=あなたのBacklog APIキー
export OPENAI_API_KEY=あなたのOpenAI APIキー
```

2. スクリプトを実行:

```bash
python backlog_wiki.py
```

3. プロンプトに従って操作:
   - プロジェクト一覧から選択
   - Wikiのインデックス作成が完了すると質問モードになります
   - 質問を入力して情報を検索
   - 終了するには 'exit' と入力

## 技術的な詳細

- `BacklogClient`: Backlog APIとの通信を行うクラス
- `RAGSystem`: RAGシステムの機能を提供するクラス
  - Wikiコンテンツの処理・分割
  - ベクトルDBの管理
  - 検索・質問応答機能の提供

## 注意点

- APIキーは安全に管理してください
- 大量のWikiを処理する場合、OpenAI APIの利用料金が発生する可能性があります
- ベクトルDBはローカルに保存されます（デフォルトでは `./chroma_db` ディレクトリ）

## ライセンス

MITライセンス
