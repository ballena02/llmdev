import os
from dotenv import load_dotenv
import tiktoken
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Annotated
from typing_extensions import TypedDict

# 環境変数を読み込む
load_dotenv(".env")
os.environ['OPENAI_API_KEY'] = os.environ['API_KEY']

# 使用するモデル名
MODEL_NAME = "gpt-4o-mini" 

# MemorySaverインスタンスの作成
memory = MemorySaver()

# グラフを保持する変数の初期化
graph = None

# ===== Stateクラスの定義 =====
# Stateクラス: メッセージのリストを保持する辞書型
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ===== インデックスの構築 =====
def create_index(persist_directory, embedding_model):
    # 実行中のスクリプトのパスを取得
    current_script_path = os.path.abspath(__file__)
    # 実行中のスクリプトが存在するディレクトリを取得
    current_directory = os.path.dirname(current_script_path)

    # テキストファイルを読込
    loader = DirectoryLoader(f'{current_directory}/data/pdf', glob="./*.pdf",   loader_cls=PyPDFLoader)
    documents = loader.load()

    # チャンクに分割
    encoding_name = tiktoken.encoding_for_model(MODEL_NAME).name
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(encoding_name)
    texts = text_splitter.split_documents(documents)

    # 新規にIndexを構築
    db = Chroma.from_documents(texts, embedding_model, persist_directory=persist_directory)
    return db

def define_tools():
    # 実行中のスクリプトのパスを取得
    current_script_path = os.path.abspath(__file__)
    # 実行中のスクリプトが存在するディレクトリを取得
    current_directory = os.path.dirname(current_script_path)

    # インデックスの保存先
    persist_directory = f'{current_directory}/chroma_db'
    # エンベディングモデル
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists(persist_directory):
        try:
            # ストレージから復元
            db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
            print("既存のインデックスを復元しました。")
        except Exception as e:
            print(f"インデックスの復元に失敗しました: {e}")
            db = create_index(persist_directory, embedding_model)
    else:
        print(f"インデックスを新規作成します。")
        db = create_index(persist_directory, embedding_model)

    # Retrieverの作成
    retriever = db.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_company_rules",
        "Search and return company rules",
    )

    # Web検索ツール
    tavily_tool = TavilySearchResults(max_results=2)

    return [retriever_tool, tavily_tool]

# ===== グラフの構築 =====
def build_graph(model_name, memory):
    """
    グラフのインスタンスを作成し、ツールノードやチャットボットノードを追加します。
    モデル名とメモリを使用して、実行可能なグラフを作成します。
    """
    # グラフのインスタンスを作成
    graph_builder = StateGraph(State)

    # ツールノードの作成
    tools = define_tools()
    tool_node = ToolNode(tools)
    graph_builder.add_node("tools", tool_node)

    # チャットボットノードの作成
    llm = ChatOpenAI(model_name=model_name)
    llm_with_tools = llm.bind_tools(tools)
    
    # チャットボットの実行方法を定義
    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    graph_builder.add_node("chatbot", chatbot)

    # 実行可能なグラフの作成
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")
    
    return graph_builder.compile(checkpointer=memory)

# ===== グラフを実行する関数 =====
def stream_graph_updates(graph: StateGraph, user_message: str, thread_id):
    """
    ユーザーからのメッセージを元に、グラフを実行し、チャットボットの応答をストリーミングします。
    """
    response = graph.invoke(
        {"messages": [("user", user_message)]},
        {"configurable": {"thread_id": thread_id}},
        stream_mode="values"
    )
    return response["messages"][-1].content

# ===== 応答を返す関数 =====
def get_bot_response(user_message, memory, thread_id):
    """
    ユーザーのメッセージに基づき、ボットの応答を取得します。
    初回の場合、新しいグラフを作成します。
    """
    global graph
    # グラフがまだ作成されていない場合、新しいグラフを作成
    if graph is None:
        graph = build_graph(MODEL_NAME, memory)

    # グラフを実行してボットの応答を取得
    return stream_graph_updates(graph, user_message, thread_id)

# ===== メッセージの一覧を取得する関数 =====
def get_messages_list(memory, thread_id):
    """
    メモリからメッセージ一覧を取得し、ユーザーとボットのメッセージを分類します。
    """
    messages = []
    # メモリからメッセージを取得
    memories = memory.get({"configurable": {"thread_id": thread_id}})['channel_values']['messages']
    for message in memories:
        if isinstance(message, HumanMessage):
            # ユーザーからのメッセージ
            messages.append({'class': 'user-message', 'text': message.content.replace('\n', '<br>')})
        elif isinstance(message, AIMessage) and message.content != "":
            # ボットからのメッセージ（最終回答）
            messages.append({'class': 'bot-message', 'text': message.content.replace('\n', '<br>')})
    return messages
