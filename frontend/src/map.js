import React, { Component } from 'react';
import ReactDOM from 'react-dom'
import Header from './Header'
import './menu.css'
import Menu from './Menu'
import Map from 'pigeon-maps'
import Marker from 'pigeon-marker'
import { GeolocatedProps, geolocated } from 'react-geolocated';
 
interface IDemoProps {
  label: string;
}

class MyMap extends Component<IDemoProps & GeolocatedProps> { 
  render() {
        return (
          <div>
             < Header text="Map"/>
            <Menu />
          <Map center={[this.props.coords && this.props.coords.latitude, this.props.coords && this.props.coords.longitude]} zoom={15} width={document.innerWidth} height={400}>
          <Marker anchor={[this.props.coords && this.props.coords.latitude, this.props.coords && this.props.coords.longitude]} payload={1} onClick={({ event, anchor, payload }) => {}} />
      
          {/* <Overlay anchor={[50.879, 4.6997]} offset={[120, 79]}>
            <img src='pigeon.jpg' width={240} height={158} alt='' />
          </Overlay> */}
        </Map>
        <div>
        label: {this.props.label}
        lattitude: {this.props.coords && this.props.coords.latitude}
        longitude: {this.props.coords && this.props.coords.longitude}
      </div>
        </div>
        );
    }
  }



export default geolocated()(MyMap);
