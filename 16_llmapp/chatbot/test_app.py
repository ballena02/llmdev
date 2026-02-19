import pytest
from flask import session
from chatbot.app import app
from chatbot.graph import memory, get_messages_list

USER_MESSAGE_1 = "1たす2は？"
USER_MESSAGE_2 = "東京駅のイベントの検索結果を教えて"

@pytest.fixture
def client():
    """
    Flaskテストクライアントを作成。
    """
    app.config['TESTING'] = True
    app.config['SECRET_KEY'] = 'your_secret_key'  # セッション用の秘密鍵を設定
    client = app.test_client()
    with client.session_transaction() as session:
        session.clear()  # セッションをクリアして初期化
    yield client

def test_index_get_request(client):
    """
    GETリクエストで初期画面が正しく表示されるかをテスト。
    """
    response = client.get('/')
    assert response.status_code == 200, "GETリクエストに対してステータスコード200を返すべきです。"
    assert b"<form" in response.data, "HTMLにフォーム要素が含まれている必要があります。"
    assert memory.storage == {}, "GETリクエストでメモリが初期化されるべきです。"

def test_index_post_request(client):
    """
    POSTリクエストでボットの応答が正しく返されるかをテスト。
    """
    with client.session_transaction() as session:
        thread_id = session.get('thread_id')
        assert thread_id is None, "初期状態ではセッションにthread_idが設定されていないはずです。"

    response = client.post('/', data={'user_message': USER_MESSAGE_1})
    assert response.status_code == 200, "POSTリクエストに対してステータスコード200を返すべきです。"
    decoded_data = response.data.decode('utf-8')  # バイト文字列をデコード
    assert "1たす2" in decoded_data, "ユーザーの入力がHTML内に表示されるべきです。"
    assert "3" in decoded_data, "ボットの応答が正しくHTML内に表示されるべきです。"

    with client.session_transaction() as session:
        thread_id = session.get('thread_id')
        assert thread_id is not None, "POSTリクエスト後にはセッションにthread_idが設定されているべきです。"

def test_memory_persistence_with_session(client):
    """
    複数のPOSTリクエストでメモリがセッションごとに保持されるかをテスト。
    """
    client.post('/', data={'user_message': USER_MESSAGE_1})
    client.post('/', data={'user_message': USER_MESSAGE_2})

    with client.session_transaction() as session:
        thread_id = session.get('thread_id')
        assert thread_id is not None, "セッションにはthread_idが設定されている必要があります。"

    messages = get_messages_list(memory, thread_id)
    assert len(messages) >= 2, "メモリに2つ以上のメッセージが保存されるべきです。"
    assert any("1たす2" in msg['text'] for msg in messages if msg['class'] == 'user-message'), "メモリに最初のユーザーメッセージが保存されるべきです。"
    assert any("東京駅" in msg['text'] for msg in messages if msg['class'] == 'user-message'), "メモリに2番目のユーザーメッセージが保存されるべきです。"

def test_clear_endpoint(client):
    """
    /clearエンドポイントがセッションとメモリを正しくリセットするかをテスト。
    """
    client.post('/', data={'user_message': USER_MESSAGE_1})

    with client.session_transaction() as session:
        thread_id = session.get('thread_id')
        assert thread_id is not None, "POSTリクエスト後にはセッションにthread_idが設定されているべきです。"

    response = client.post('/clear')
    assert response.status_code == 200, "POSTリクエストに対してステータスコード200を返すべきです。"
    assert b"<form" in response.data, "HTMLにフォーム要素が含まれている必要があります。"

    with client.session_transaction() as session:
        thread_id = session.get('thread_id')
        assert thread_id is None, "/clearエンドポイント後にはセッションにthread_idが設定されていないべきです。"

    # メモリがクリアされているか確認
    cleared_messages = memory.get({"configurable": {"thread_id": thread_id}})
    assert cleared_messages is None, "メモリは/clearエンドポイント後にクリアされるべきです。"