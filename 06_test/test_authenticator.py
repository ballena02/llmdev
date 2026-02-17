# test_authenticator.py
import pytest
from authenticator import Authenticator


def test_register_success():
    auth = Authenticator()
    auth.register("alice", "password123")
    assert auth.users["alice"] == "password123"


def test_register_existing_user_raises():
    auth = Authenticator()
    auth.register("alice", "password123")

    with pytest.raises(ValueError) as excinfo:
        auth.register("alice", "newpass")

    assert str(excinfo.value) == "エラー: ユーザーは既に存在します。"


def test_login_success():
    auth = Authenticator()
    auth.register("alice", "password123")

    result = auth.login("alice", "password123")
    assert result == "ログイン成功"


def test_login_wrong_password_raises():
    auth = Authenticator()
    auth.register("alice", "password123")

    with pytest.raises(ValueError) as excinfo:
        auth.login("alice", "wrongpass")

    assert str(excinfo.value) == "エラー: ユーザー名またはパスワードが正しくありません。"
