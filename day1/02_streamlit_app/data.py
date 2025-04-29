# data.py
import streamlit as st
from datetime import datetime
from database import save_to_db, get_db_count # DB操作関数をインポート

# サンプルデータのリスト
SAMPLE_QUESTIONS_DATA = [
    {
        "question": "Pythonのリスト内包表記とは何ですか？",
        "answer": "リスト内包表記は、既存のリストから新しいリストを作成するためのPythonの構文です。通常のfor文よりも簡潔に記述でき、パフォーマンスも向上する場合があります。",
        "correct_answer": "Pythonのリスト内包表記は、リストを簡潔に作成するための構文で、`[expression for item in iterable if condition]`の形式で書きます。通常のforループよりも短く書けて、実行速度も速い場合があります。",
        "feedback": "部分的に正確: 基本的な説明は正しいですが、具体的な構文例が示されていません",
        "is_correct": 0.5,
        "response_time": 1.2,
        "share_internally": False
    },
    {
        "question": "機械学習における過学習とは？",
        "answer": "過学習（オーバーフィッティング）とは、機械学習モデルが訓練データに対して過度に適合し、新しいデータに対する汎化性能が低下する現象です。",
        "correct_answer": "過学習（オーバーフィッティング）は、モデルがトレーニングデータに過度に適合し、未知のデータに対する予測性能が低下する現象です。モデルが訓練データのノイズまで学習してしまうことが原因です。",
        "feedback": "正確: 過学習の本質をよく捉えています",
        "is_correct": 1.0, # 整数ではなく浮動小数点数で統一
        "response_time": 1.5,
        "share_internally": False
    },
    # ... (他のサンプルデータも同様に追加) ...
    {
        "question": "量子コンピュータの基本原理は？",
        "answer": "量子コンピュータは量子力学の原理に基づいて動作します。従来のビットの代わりに量子ビット（キュービット）を使用し、重ね合わせと量子もつれの特性により並列計算を実現します。",
        "correct_answer": "量子コンピュータは量子力学的現象を利用した計算機で、従来のビットではなく量子ビット（キュービット）を使用します。キュービットは重ね合わせ状態をとることができ、複数の状態を同時に表現できます。また量子もつれにより、従来のコンピュータでは困難な特定の問題を効率的に解くことができます。",
        "feedback": "部分的に正確: 基本概念は正しいですが、詳細な説明が不足しています",
        "is_correct": 0.5,
        "response_time": 2.1,
        "share_internally": False
    },
    {
        "question": "Streamlitとは何ですか？",
        "answer": "Streamlitは、Pythonで書かれたデータサイエンスやAIアプリケーションを簡単に作成するためのオープンソースフレームワークです。数行のコードでインタラクティブなWebアプリを作成できます。",
        "correct_answer": "Streamlitは、データサイエンティストやAIエンジニアがPythonを使って簡単にWebアプリケーションを構築できるフレームワークです。少ないコード量で、インタラクティブなダッシュボードやデータ可視化アプリケーションを作成できます。",
        "feedback": "正確: Streamlitの基本概念と利点をよく説明しています",
        "is_correct": 1.0,
        "response_time": 0.9,
        "share_internally": False
    },
    {
        "question": "ブロックチェーンの仕組みを説明してください",
        "answer": "ブロックチェーンは、分散型台帳技術の一つで、データをブロックに格納し、暗号技術でリンクして改ざん防止を実現します。各ブロックには前のブロックのハッシュ値が含まれ、チェーン状に連結されています。",
        "correct_answer": "ブロックチェーンは分散型台帳技術で、データブロックが暗号学的にリンクされた構造です。各ブロックには取引データとタイムスタンプ、前ブロックのハッシュ値が含まれます。分散型ネットワークでコンセンサスアルゴリズムにより検証され、改ざんが極めて困難なシステムを実現しています。",
        "feedback": "部分的に正確: 基本的な説明はありますが、コンセンサスメカニズムについての言及がありません",
        "is_correct": 0.5,
        "response_time": 1.8,
        "share_internally": False
    },
        {
        "question": "ディープラーニングとは何ですか？",
        "answer": "ディープラーニングは、複数の層からなるニューラルネットワークを用いた機械学習手法です。画像認識や自然言語処理など複雑なタスクに優れています。",
        "correct_answer": "ディープラーニングは多層ニューラルネットワークを使用した機械学習の一種で、特徴抽出を自動的に行う能力があります。画像認識、自然言語処理、音声認識などの複雑なタスクで革命的な成果を上げており、大量のデータと計算リソースを活用して従来の手法を超える性能を実現しています。",
        "feedback": "部分的に正確: 基本的な定義は正しいですが、詳細な説明が不足しています",
        "is_correct": 0.5,
        "response_time": 1.3,
        "share_internally": False
    },
    {
        "question": "SQLインジェクションとは何ですか？",
        "answer": "SQLインジェクションは、Webアプリケーションの脆弱性を悪用して不正なSQLクエリを実行させる攻撃手法です。ユーザー入力を適切に検証・サニタイズしないことで発生します。",
        "correct_answer": "SQLインジェクションは、Webアプリケーションのセキュリティ脆弱性を悪用した攻撃手法で、攻撃者がユーザー入力フィールドを通じて悪意のあるSQLコードを挿入し、データベースに不正なクエリを実行させます。これにより、データの漏洩、改ざん、削除などの被害が生じる可能性があります。防止策としては、パラメータ化クエリの使用、入力のバリデーション、最小権限の原則などがあります。",
        "feedback": "正確: SQLインジェクションの本質と発生メカニズムをよく説明しています",
        "is_correct": 1.0,
        "response_time": 1.6,
        "share_internally": False
    },
    {
        "question": "NFTとは何ですか？",
        "answer": "NFT（Non-Fungible Token）は、代替不可能なトークンで、デジタルアセットの所有権を証明するためのブロックチェーン技術です。デジタルアート、コレクティブル、音楽などに利用されています。",
        "correct_answer": "NFT（Non-Fungible Token、非代替性トークン）はブロックチェーン上に記録された固有の識別子を持つデジタル資産です。通常の暗号通貨と異なり、各NFTは独自の価値を持ち、交換不可能です。デジタルアート、音楽、ゲーム内アイテム、バーチャル不動産など様々なデジタル資産の所有権証明や取引に利用されています。",
        "feedback": "正確: NFTの基本概念とユースケースを明確に説明しています",
        "is_correct": 1.0,
        "response_time": 1.4,
        "share_internally": False
    },
    {
        "question": "Pythonのデコレータとは何ですか？",
        "answer": "デコレータは、関数やメソッドを修飾するための構文で、@記号を使用します。関数の機能を変更したり拡張したりするための便利な方法です。",
        "correct_answer": "Pythonのデコレータは、既存の関数やメソッドを修飾して機能を拡張するための構文です。@記号を使用して関数定義の前に配置します。デコレータは高階関数で、別の関数を引数として受け取り、新しい関数を返します。ロギング、認証、キャッシングなど、コードの重複を避けながら横断的関心事を実装するのに役立ちます。",
        "feedback": "部分的に正確: 基本的な説明はありますが、デコレータが高階関数であることや具体的な使用例の説明が不足しています",
        "is_correct": 0.5,
        "response_time": 1.2,
        "share_internally": False
    },
    {
        "question": "コンテナ技術とは何ですか？",
        "answer": "コンテナ技術は、アプリケーションとその依存関係をパッケージ化し、異なる環境で一貫して実行できるようにする軽量な仮想化技術です。",
        "correct_answer": "コンテナ技術は、アプリケーションとその依存関係（ライブラリ、バイナリなど）を一つのパッケージにカプセル化する軽量な仮想化技術です。コンテナは仮想マシンよりも軽量で起動が速く、ホストOSのカーネルを共有します。Dockerが代表的なコンテナプラットフォームで、アプリケーションの開発、テスト、デプロイメントを効率化し、「どこでも同じように動作する」環境を提供します。",
        "feedback": "部分的に正確: 基本的な説明はありますが、仮想マシンとの違いやDockerなどの具体例の説明が不足しています",
        "is_correct": 0.5,
        "response_time": 1.1,
        "share_internally": False
    }
]


def create_sample_evaluation_data():
    """定義されたサンプルデータをデータベースに保存する"""
    try:
        count_before = get_db_count()
        added_count = 0
        # 各サンプルをデータベースに保存
        for item in SAMPLE_QUESTIONS_DATA:
            # save_to_dbが必要な引数のみ渡す
            save_to_db(
                question=item["question"],
                answer=item["answer"],
                feedback=item["feedback"],
                correct_answer=item["correct_answer"],
                is_correct=item["is_correct"],
                response_time=item["response_time"],
                share_internally=item["share_internally"]
            )
            added_count += 1

        count_after = get_db_count()
        st.success(f"{added_count} 件のサンプル評価データが正常に追加されました。(合計: {count_after} 件)")

    except Exception as e:
        st.error(f"サンプルデータの作成中にエラーが発生しました: {e}")
        print(f"エラー詳細: {e}") # コンソールにも出力

def ensure_initial_data():
    """データベースが空の場合に初期サンプルデータを投入する"""
    if get_db_count() == 0:
        st.info("データベースが空です。初期サンプルデータを投入します。")
        create_sample_evaluation_data()