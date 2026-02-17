class Authenticator:
    def __init__(self):
        self.users = {}  # ユーザー情報を格納（インスタンス変数）

    # ユーザーの登録
    def register(self, username, password):
        if username in self.users:
            raise ValueError("エラー: ユーザーは既に存在します。")
        else:
            self.users[username] = password

    # ログイン
    def login(self, username, password):
        if self.users.get(username) == password:
            return "ログイン成功"
        else:
            raise ValueError("エラー: ユーザー名またはパスワードが正しくありません。")