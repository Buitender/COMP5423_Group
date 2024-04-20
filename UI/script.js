
// 获取元素
const messageInput = document.getElementById('message-input');
const submitBtn = document.getElementById('submit-btn');
const chatDisplayBart = document.querySelector('.chat-display.bart');
const chatDisplayApi = document.querySelector('.chat-display.api');
const bartBtn = document.getElementById('bart-btn');
const apiBtn = document.getElementById('api-btn');

bartBtn.addEventListener('click', useBart);
apiBtn.addEventListener('click', useApi);

let type = "bart";

function useBart() {
    type = "bart";
    bartBtn.style.backgroundColor = "#355f86";
    apiBtn.style.backgroundColor = "#428bca";
    chatDisplayBart.style.display = "block";
    chatDisplayApi.style.display = "none";
}

function useApi() {
    type = "api"
    bartBtn.style.backgroundColor = "#428bca";
    apiBtn.style.backgroundColor = "#355f86"
    chatDisplayBart.style.display = "none";
    chatDisplayApi.style.display = "block";
}


// 添加事件监听器
submitBtn.addEventListener('click', sendMessage);
messageInput.addEventListener('keyup', function (event){
    if (event.key === 'Enter'){
        event.preventDefault()
        sendMessage();
    }
})

// 定义函数：发送消息
function sendMessage() {
    // 获取输入的消息内容
    const message = messageInput.value;
    if (message.trim()==''){
        return
    }
    // 创建一个新的消息元素
    const newMessage = document.createElement('div');
    const msgDiv = document.createElement('div')
    msgDiv.innerHTML = message
    msgDiv.style.background = '#428bca'
    // msgDiv.style.height = '20px'
    msgDiv.style.marginRight = '5px'
    msgDiv.style.alignContent = 'center'
    msgDiv.style.padding = '5px'
    msgDiv.style.wordBreak = 'break-word'
    // 设置消息内容和样式
    newMessage.appendChild(msgDiv)
    newMessage.style.color = 'white';
    newMessage.style.padding = '5px 10px';
    newMessage.style.marginBottom = '10px';
    // 获取用户头像的URL
    const avatarUrl = './chatbot.png';
    // 创建并设置头像元素
    const avatar = document.createElement('img');
    avatar.src = avatarUrl;
    avatar.width = 40;
    avatar.height = 40;
    // 将头像元素添加到消息元素的前面
    newMessage.appendChild(avatar);
    // newMessage.style['text-align'] = 'right';
    newMessage.style['display'] = 'inline-flex';
    newMessage.style['justify-content'] = 'right'
    newMessage.style['align-items'] = 'center'
    newMessage.style['width'] = '98%'

    // 将消息添加到对话展示框中
    if (type === "bart"){
        chatDisplayBart.appendChild(newMessage)
    }else {
        chatDisplayApi.appendChild(newMessage)
    }
    newMessage.scrollIntoView({behavior: 'smooth', block: 'end'})
    send(message)
    // 清空输入框
    messageInput.value = '';
}


function receive(message){
    const newMessage = document.createElement('div');
    const msgDiv = document.createElement('div')
    msgDiv.innerHTML = message
    msgDiv.style.background = '#428bca'
    msgDiv.style.marginRight = '5px'
    msgDiv.style.alignContent = 'center'
    msgDiv.style.padding = '5px'
    // 设置消息内容和样式
    newMessage.appendChild(msgDiv)
    newMessage.style.color = 'white';
    newMessage.style.padding = '5px 10px';
    newMessage.style.marginBottom = '10px';
    // 获取用户头像的URL
    const avatarUrl = './chatbot.png';
    // 创建并设置头像元素
    const avatar = document.createElement('img');
    avatar.src = avatarUrl;
    avatar.width = 40;
    avatar.height = 40;

    // 设置消息的样式
    newMessage.style.color = 'white';
    // 将头像元素添加到消息元素的前面
    newMessage.insertBefore(avatar, newMessage.firstChild);
    newMessage.style['text-align'] = 'left';
    newMessage.style['display'] = 'inline-flex';
    newMessage.style['justify-content'] = 'left'
    newMessage.style['align-items'] = 'center'
    newMessage.style['width'] = '98%'

    // 将消息添加到对话展示框中
    if (type === "bart"){
        chatDisplayBart.appendChild(newMessage)
    }else {
        chatDisplayApi.appendChild(newMessage)
    }
    newMessage.scrollIntoView({behavior: 'smooth', block: 'end'})
}

function send(msg){
    let url = "";
    if (type==="bart"){
        url = "http://127.0.0.1:5000/send2bart?message=" + msg;
    }else {
        url = "http://127.0.0.1:5000/send2api?message=" + msg;
    }
    fetch(url)
        .then(res => res.text())
        .then(data => {
            receive(data)
        })
}
