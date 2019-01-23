import React, { Component } from 'react';
import ReactDOM from 'react-dom'
import Header from './Header'
import './menu.css'
import Menu from './Menu'
import Map from 'pigeon-maps'
import Marker from 'pigeon-marker'
import PropTypes from "prop-types";
import { GeolocatedProps, geolocated } from 'react-geolocated';
import axios from 'axios';
import react from 'pigeon-marker';
 
interface IDemoProps {
  label: string;
}

class MyMap extends Component<IDemoProps & GeolocatedProps> { 
  state = {
      data: [],
      loaded: false,
      placeholder: "Loading..."
    };

  fetchData() {
    // var endPoint = 'https://api.opencagedata.com/geocode/v1/json?key=ceaef58b33f442e790f75f602065215a&q='+this.props.coords && this.props.coords.latitude +'%2C' +this.props.coords && this.props.coords.longitude + '&pretty=1';
    var endPoint = 'https://api.opencagedata.com/geocode/v1/json?key=ceaef58b33f442e790f75f602065215a&q=35.6961%2C51.4231&pretty=1';
    axios.get(endPoint)
      .then(res=>{
        var result='';
        var address=res.data.results[0].components;
        console.log(address);
        if (address.country != undefined){
          console.log(address.country);
          result += address.country+',';
        }
        if (address.county != undefined){
          console.log(address.county);
          result+=address.county+',';
        }
        if (address.neighbourhood != undefined){
          console.log(address.neighbourhood);
          result+=address.neighbourhood+','
        }
        if (address.road != undefined){
          console.log(address.road);
          result+=address.road;
        }
        ReactDOM.findDOMNode(this.refs.nothing).innerHTML = result;
      })
  }
  componentDidMount(){
    var endPoint = 'https://api.opencagedata.com/geocode/v1/json?key=ceaef58b33f442e790f75f602065215a&q=35.6961%2C51.4231&pretty=1';
    axios.get(endPoint)
    .then(res=>{
      var result='';
      var address=res.data.results[0].components;
      console.log(address);
      if (address.country !== undefined){
        console.log(address.country);
        result += address.country+',';
      }
      if (address.county !== undefined){
        console.log(address.county);
        result+=address.county+',';
      }
      if (address.neighbourhood !== undefined){
        console.log(address.neighbourhood);
        result+=address.neighbourhood+','
      }
      if (address.road !== undefined){
        console.log(address.road);
        result+=address.road;
      }
      ReactDOM.findDOMNode(this.refs.nothing).innerHTML = result;
    });
    setInterval(this.fetchData, 5000);
  }
  render() {
        return (
          <div>
             < Header text="Map"/>
            <Menu />
          <Map center={[this.props.coords && this.props.coords.latitude, this.props.coords && this.props.coords.longitude]} zoom={15} width={document.innerWidth} height={400}>
          <Marker anchor={[this.props.coords && this.props.coords.latitude, this.props.coords && this.props.coords.longitude]} payload={1} onClick={({ event, anchor, payload }) => {}} />
        </Map>
        <div>
        label: {this.props.label}
        lattitude: {this.props.coords && this.props.coords.latitude}
        longitude: {this.props.coords && this.props.coords.longitude}
      </div>
      <div ref="nothing">nothing</div>
        </div>
        );
    }
  }



export default geolocated()(MyMap);
