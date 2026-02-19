const display = document.getElementById('message-display');
const input = document.getElementById('user-input');
const btn = document.getElementById('send-btn');
const hero = document.getElementById('hero-section');
const ttsToggle = document.getElementById('tts-toggle');
const moodSelector = document.getElementById('mood-selector');

let messageHistory = [];
let currentAudio = null;

// Auto-resize textarea
input.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

async function speakText(text) {
    if (!text) return;

    // Stop any current audio
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
    }

    const speakerId = moodSelector.value;

    try {
        const response = await fetch('/api/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text, speaker: speakerId })
        });

        if (response.status === 503) {
            alert("VOICEVOX engine is not running. Please start it on port 50021 to hear the voice!");
            return;
        }
        if (!response.ok) throw new Error('TTS failed');

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        currentAudio = new Audio(url);
        currentAudio.play();
    } catch (error) {
        console.error('TTS Playback error:', error);
    }
}

function appendMessage(role, content, duration = null) {
    if (hero) display.classList.add('has-messages');

    const row = document.createElement('div');
    row.className = `message-row ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    if (role === 'user') {
        avatar.innerHTML = `<img src="/static/img/userProfile.jpg" alt="U">`;
    } else if (role === 'assistant') {
        avatar.innerHTML = `<img src="/static/img/gptProfile.png" alt="G">`;
    } else {
        avatar.innerText = 'S';
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

    // AI Actions Menu
    if (role === 'assistant') {
        const actions = document.createElement('div');
        actions.className = 'message-actions';

        const copyBtn = createActionButton('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>', 'Copy');
        copyBtn.onclick = () => {
            navigator.clipboard.writeText(bubble.innerText);
            copyBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
            setTimeout(() => copyBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>', 2000);
        };

        const voiceBtn = createActionButton('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>', 'Read aloud');
        voiceBtn.onclick = () => speakText(bubble.innerText);

        const upBtn = createActionButton('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>', 'Good response');
        const downBtn = createActionButton('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"/></svg>', 'Bad response');
        const shareBtn = createActionButton('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><polyline points="16 6 12 2 8 6"/><line x1="12" y1="2" x2="12" y2="15"/></svg>', 'Share');
        const regenBtn = createActionButton('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 4v6h-6"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>', 'Regenerate');

        actions.appendChild(copyBtn);
        actions.appendChild(upBtn);
        actions.appendChild(downBtn);
        actions.appendChild(shareBtn);
        actions.appendChild(regenBtn);
        actions.appendChild(voiceBtn);
        textDiv.appendChild(actions);
    }

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

function createActionButton(svg, title) {
    const btn = document.createElement('button');
    btn.className = 'action-icon-btn';
    btn.innerHTML = svg;
    btn.title = title;
    return btn;
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

        const finalContent = fullResponse.trim();
        messageHistory.push({ role: 'assistant', content: finalContent });

        // Auto-speak if toggled on
        if (ttsToggle.checked) {
            speakText(finalContent);
        }

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

// Sidebar Toggle for Mobile
const menuToggle = document.getElementById('menu-toggle');
const sidebar = document.querySelector('.sidebar');
const sidebarOverlay = document.getElementById('sidebar-overlay');

function toggleSidebar() {
    sidebar.classList.toggle('open');
    if (sidebarOverlay) sidebarOverlay.classList.toggle('active');
}

if (menuToggle) {
    menuToggle.addEventListener('click', toggleSidebar);
}

if (sidebarOverlay) {
    sidebarOverlay.addEventListener('click', toggleSidebar);
}

// Close sidebar when clicking 'New Chat' on mobile
const newChatBtn = document.querySelector('.new-chat-btn');
if (newChatBtn) {
    newChatBtn.addEventListener('click', () => {
        if (window.innerWidth <= 768) {
            sidebar.classList.remove('open');
            if (sidebarOverlay) sidebarOverlay.classList.remove('active');
        }
    });
}
