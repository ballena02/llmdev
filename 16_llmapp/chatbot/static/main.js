window.onload = function() {
  // チャットボックスを取得
  const chatBox = document.getElementById('chat-box');
  
  // チャットボックスのスクロールを一番下に設定
  chatBox.scrollTop = chatBox.scrollHeight;

  // Ctrl + Enterでフォームを送信
  const form = document.getElementById('chat-form');
  const textarea = document.getElementById('user-input');

  textarea.addEventListener('keydown', function(event) {
      // Ctrl + Enterが押された場合
      if (event.ctrlKey && event.key === 'Enter') {
          event.preventDefault();  // デフォルトの動作（改行など）を防止
          form.submit();  // フォームを送信
      }
  });
}