import '../styles/chat.scss';
import { ReactComponent as Search } from "../assets/img/search-icon.svg";
import { ReactComponent as Profile } from "../assets/img/profile-img.svg";
import axios from 'axios';
import { useState } from 'react';

const Chat = () => {
    const [input, setInput] = useState('');
    const [contents, setContents] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const clickQuestionButton = async () => {
        setIsLoading(true);
        setContents((contents) => [...contents, {
            name: 'User',
            content: input
        }])
        const answer = await axios.post('http://127.0.0.1:8000/', {
            question: input
        });

        setContents((contents) => [...contents, {
            name: 'KBOT',
            content: String(answer.data.answer)
        }])
        setInput('');
        setIsLoading(false);
    };
    const changeInput = (event) => {
        setInput(event.target.value);
    };
    return(
        <div className="chat">
            <div className="chat__search">
                <input
                    className="chat__search-input"
                    placeholder="야구의 궁금한 모든 것을 물어보세요."
                    value={input}
                    onChange={changeInput}
                />
                <button
                    className="chat__search-button"
                    onClick={clickQuestionButton}
                >
                    <Search/>
                </button>
            </div>
            <div className="chat__main">
                <ul className="chat__main-ul">
                    {
                        contents.map((content) => {
                            return (
                                <li className="chat__main-ul-li" key={content}>
                                    <div className="chat__main-ul-li-header">
                                        <Profile className="chat__main-ul-li-header-img"/>
                                        <h4 className="chat__main-ul-li-header-user">
                                            {content.name}
                                        </h4>
                                    </div>
                                    <p className="chat__main-ul-li-content">
                                        { content.content }
                                    </p>
                                </li>
                            )
                        })
                    }
                </ul>
            </div>
            {
                isLoading &&
                <div className='loading'>
                    <svg class="spinner" width="65px" height="65px" viewBox="0 0 66 66" xmlns="http://www.w3.org/2000/svg">
                        <circle class="path" fill="none" stroke-width="6" stroke-linecap="round" cx="33" cy="33" r="30"></circle>
                    </svg>
                    <p className='loading__info'>답변 생성중입니다. 잠시만 기다려주세요.</p>
                </div>
            }
        </div>
    )
};

export default Chat;