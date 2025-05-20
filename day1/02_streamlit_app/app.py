# app.py
import streamlit as st
import ui  # UIモジュール
import llm  # LLMモジュール
import database  # データベースモジュール
import metrics  # 評価指標モジュール
import data  # データモジュール
import torch
from transformers import pipeline
from config import MODEL_NAME
from huggingface_hub import HfFolder
import os


# CSSファイルの読み込み
def load_css():
    css_file = os.path.join(os.path.dirname(__file__), "style.css")
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# --- アプリケーション設定 ---
# st.set_page_config(page_title="Gemma Chatbot", layout="wide")
st.set_page_config(
    page_title="ジョブクラウン社内チャットボットサイト",
    layout="wide",
    page_icon="👔",  # 企業用のアイコン
    initial_sidebar_state="expanded",
)

# --- 初期化処理 ---
# NLTKデータのダウンロード（初回起動時など）
metrics.initialize_nltk()

# データベースの初期化（テーブルが存在しない場合、作成）
database.init_db()

# データベースが空ならサンプルデータを投入
data.ensure_initial_data()


# LLMモデルのロード（キャッシュを利用）
# モデルをキャッシュして再利用
@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}")  # 使用デバイスを表示
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,
        )
        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return pipe
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error(
            "GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。"
        )
        return None


pipe = llm.load_model()

# --- Streamlit アプリケーション ---

# CSSの適用
load_css()

# カスタムテーマの設定
st.markdown(
    """
    <style>
    :root {
        --primary-color: #2c3e50;
        --background-color: #ffffff;
        --secondary-background-color: #f8f9fa;
        --text-color: #2c3e50;
        --font: "Helvetica Neue", sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# st.title("🤖 Gemma 2 Chatbot with Feedback")
st.title("ジョブクラウン社内チャットボット")
# st.write("Gemmaモデルを使用したチャットボットです。回答に対してフィードバックを行えます。")
st.write("社内の質問や業務サポートを行うAIアシスタントです。")
st.markdown(
    """
- 社内規定や手続きに関する質問
- 業務効率化のアドバイス
- 社内システムの使用方法
などについてお答えします。
"""
)
st.markdown("---")

# --- サイドバー ---
st.sidebar.title("ナビゲーション")
# セッション状態を使用して選択ページを保持
if "page" not in st.session_state:
    st.session_state.page = "チャット"  # デフォルトページ

page = st.sidebar.radio(
    "ページ選択",
    ["チャット", "履歴閲覧", "サンプルデータ管理"],
    key="page_selector",
    index=["チャット", "履歴閲覧", "サンプルデータ管理"].index(
        st.session_state.page
    ),  # 現在のページを選択状態にする
    on_change=lambda: setattr(
        st.session_state, "page", st.session_state.page_selector
    ),  # 選択変更時に状態を更新
)


# --- メインコンテンツ ---
if st.session_state.page == "チャット":
    if pipe:
        ui.display_chat_page(pipe)
    else:
        st.error("チャット機能を利用できません。モデルの読み込みに失敗しました。")
elif st.session_state.page == "履歴閲覧":
    ui.display_history_page()
elif st.session_state.page == "サンプルデータ管理":
    ui.display_data_page()

# --- フッターなど（任意） ---
st.sidebar.markdown("---")
# st.sidebar.info("開発者: [Your Name]")
st.sidebar.info("開発者: 吉川耕平")
