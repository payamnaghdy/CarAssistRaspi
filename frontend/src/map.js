import React, { Component } from 'react';
import ReactDOM from 'react-dom'
import Header from './Header'
import Map from 'pigeon-maps'
// import Marker from 'pigeon-marker'
// import Overlay from 'pigeon-overlay'
import { slide as Menu } from 'react-burger-menu'
class MyMap extends Component { 
    render() {
          return (
            <div>
               < Header text="Map"/>
               <Menu  width={ 100 }>
            <a id="home" className="menu-item" href="/">Home</a>
            <a id="about" className="menu-item" href="/map">Map</a>
            <a id="contact" className="menu-item" href="/speed">speed</a>
             <a onClick={ this.showSettings } className="menu-item--small" href="">Settings</a>
            </Menu>
            <Map center={[32.6546, 51.6680]} zoom={12} width={300} height={400}>
            {/* <Marker anchor={[50.874, 4.6947]} payload={1} onClick={({ event, anchor, payload }) => {}} /> */}
        
            {/* <Overlay anchor={[50.879, 4.6997]} offset={[120, 79]}>
              <img src='pigeon.jpg' width={240} height={158} alt='' />
            </Overlay> */}
          </Map>
          </div>
          );
      }
    }
  
  
  
  export default MyMap;
  