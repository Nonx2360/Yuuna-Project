const chatWindow = document.getElementById('chat-window');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

let messageHistory = [];

function appendMessage(role, content) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `msg ${role}`;

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = content;

    msgDiv.appendChild(bubble);
    chatWindow.appendChild(msgDiv);

    // Scroll to bottom
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    // Add user message to UI
    appendMessage('user', text);
    userInput.value = '';

    // Add to history
    messageHistory.push({ role: 'user', content: text });

    // Show typing placeholder (simulated)
    const typingMsg = document.createElement('div');
    typingMsg.className = 'msg assistant typing';
    typingMsg.innerHTML = '<div class="bubble">...</div>';
    chatWindow.appendChild(typingMsg);
    chatWindow.scrollTop = chatWindow.scrollHeight;

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ messages: messageHistory })
        });

        const data = await response.json();

        // Remove typing placeholder
        chatWindow.removeChild(typingMsg);

        if (data.response) {
            appendMessage('assistant', data.response);
            messageHistory.push({ role: 'assistant', content: data.response });
        } else if (data.error) {
            appendMessage('system', `Error: ${data.error}`);
        }
    } catch (error) {
        chatWindow.removeChild(typingMsg);
        appendMessage('system', 'Sorry, I couldn\'t reach the server.');
        console.error('Fetch error:', error);
    }
}

sendBtn.addEventListener('click', sendMessage);

userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});
