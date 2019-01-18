import React, { Component } from 'react';
import ReactDOM from 'react-dom'
class Menu extends Component { 
    render() {
          return (
            
            <nav>
            <ul class="menu">
              <li><a href="/speed" className='a'>Speed</a></li>
              <li><a href="/Map" className='a'>Map</a></li>
              <li><a href="#!" className='a'>Contact</a></li>
              <li><a href="#!" className='a'>Faq</a></li>
            </ul>
          </nav>    
          );
      }
    }
  
  
  
  export default Menu;
  