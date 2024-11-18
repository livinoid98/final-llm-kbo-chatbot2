import '../styles/header.scss';
import { ReactComponent as Profile } from "../assets/img/profile-img.svg";

const Header = () => {
    return (
        <div>
            <div className="main-banner">
                <div className="main-banner__black"></div>
                <header className="header">
                    <div className="header__logo">
                        LOGO
                    </div>
                    <ul className="header__ul">
                        <li className="header__ul-li">로그아웃</li>
                        <li className="header__ul-li">
                            <Profile
                                className="header__ul-li-profile"
                                src="../assets/img/profile-img.svg"
                            />
                        </li>
                    </ul>
                </header>
                <div className="main-banner__info">
                    <h3 className="main-banner__info-title">
                        당신이 궁금해하는{'\n'}
                        KBO의 모든 것
                    </h3>
                    <p className="main-banner__info-description">
                        실시간 스포츠 뉴스 반영 KBO{'\n'}
                        정보 알림이 kbot
                    </p>
                </div>
            </div>
        </div>
    );
};

export default Header;