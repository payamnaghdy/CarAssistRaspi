import React, { Component } from 'react';
import ReactDOM from 'react-dom'
import Header from './Header'
import './menu.css'
import Menu from './Menu'
import Map from 'pigeon-maps'

class MyMap extends Component { 
  render() {
        return (
          <div>
             < Header text="Map"/>
            <Menu />
          <Map center={[32.6546, 51.6680]} zoom={12} width={document.innerWidth} height={400}>
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
