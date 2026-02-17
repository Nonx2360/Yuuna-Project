const display = document.getElementById('message-display');
const input = document.getElementById('user-input');
const btn = document.getElementById('send-btn');
const hero = document.getElementById('hero-section');

let messageHistory = [];

// Auto-resize textarea
input.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

function appendMessage(role, content, duration = null) {
    // Hide hero if it's the first message
    if (hero) {
        display.classList.add('has-messages');
    }

    const row = document.createElement('div');
    row.className = `message-row ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    if (role === 'user') {
        avatar.innerHTML = `<img src="/static/img/userProfile.jpg" alt="U">`;
    } else {
        avatar.innerText = role === 'assistant' ? 'G' : 'S';
    }

    const textDiv = document.createElement('div');
    textDiv.className = 'text';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';

    if (role === 'user' || role === 'system') {
        bubble.textContent = content;
    } else {
        bubble.innerHTML = marked.parse(content);
    }

    textDiv.appendChild(bubble);

    if (duration !== null) {
        const timeDiv = document.createElement('div');
        timeDiv.className = 'response-time';
        timeDiv.textContent = `Response time: ${duration}s`;
        textDiv.appendChild(timeDiv);
    }

    row.appendChild(avatar);
    row.appendChild(textDiv);
    display.appendChild(row);

    display.scrollTop = display.scrollHeight;

    return { row, bubble, textDiv };
}

async function sendMessage() {
    const text = input.value.trim();
    if (!text) return;

    appendMessage('user', text);
    input.value = '';
    input.style.height = 'auto'; // Reset height
    messageHistory.push({ role: 'user', content: text });

    const { bubble: assistantBubble, textDiv: assistantTextDiv } = appendMessage('assistant', '');
    assistantBubble.textContent = '...';
    display.scrollTop = display.scrollHeight;

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ messages: messageHistory })
        });

        if (!response.ok) throw new Error('Network response was not ok');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';
        let duration = null;

        assistantBubble.textContent = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });

            if (chunk.includes('__DURATION__')) {
                const parts = chunk.split('__DURATION__');
                fullResponse += parts[0];
                duration = parts[1];

                assistantBubble.innerHTML = marked.parse(fullResponse.trim());

                const timeDiv = document.createElement('div');
                timeDiv.className = 'response-time';
                timeDiv.textContent = `Response time: ${duration}s`;
                assistantTextDiv.appendChild(timeDiv);
            } else {
                fullResponse += chunk;
                assistantBubble.innerHTML = marked.parse(fullResponse);
            }

            display.scrollTop = display.scrollHeight;
        }

        messageHistory.push({ role: 'assistant', content: fullResponse.trim() });

    } catch (error) {
        console.error('Fetch error:', error);
        assistantBubble.textContent = 'Error: Could not connect to the server.';
    }
}

btn.addEventListener('click', sendMessage);
input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});
