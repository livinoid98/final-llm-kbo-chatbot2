import '../styles/footer.scss';
import { ReactComponent as FooterLogo } from "../assets/img/footer-logo.svg";

const Footer = () => {
    return (
        <footer className='footer'>
            <FooterLogo className='footer__logo'/>
            <div className='footer__info'>
                <p className='footer__info-contact'>[02841] 서울특별시 성북구 안암로145 고려대학교 자연계캠퍼스 우정정보관 102호</p>
                <p className='footer__info-contact'>TEL : 02-3290-4931 (석사과정), 4922(계약학과) / FAX : 02-929-1917</p>
                <p className='footer__info-contact'>EMAIL : gscit@korea.ac.kr</p>
                <p className='footer__info-copyright'>개인정보 처리방침 / 개인정보처리업무위탁내역</p>
                <p className='footer__info-copyright'>Copyright (C)2016 Korea University. All Rights Reserved</p>
            </div>
        </footer>
    )
};

export default Footer;