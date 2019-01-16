import React, { Component } from 'react';
import './Header.css'
class Header extends React.Component {
    render() {
       return (
          <div>
             <h1 className='header'>{this.props.text}</h1>
          </div>
       );
    }
 }

 export default Header