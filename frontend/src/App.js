import './App.css';
import './styles/reset.scss';
import './styles/mixin.scss';
import Header from './components/Header.js';
import Chat from './components/Chat.js';
import Footer from './components/Footer.js'

function App() {
  return (
    <div className="App">
      <Header/>
      <Chat/>
      <Footer/>
    </div>
  );
}

export default App;
