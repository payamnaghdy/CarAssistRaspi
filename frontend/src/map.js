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
import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider';
import AppBar from 'material-ui/AppBar';
import RaisedButton from 'material-ui/RaisedButton';
import TextField from 'material-ui/TextField';
 
interface IDemoProps {
  label: string;
}

class MyMap extends Component<IDemoProps & GeolocatedProps> { 
  state = {
      username:'',
      password:'',
      token :'',
      token_is_available:false,
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
    if(this.state.token_is_available){
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
  }
  loginHandler(){
      console.log('salam');
      var bodyFormData = new FormData();
    bodyFormData.set('username', this.state.username);
    bodyFormData.append('password', this.state.password); 
    axios({
      method: 'post',
      url: 'http://127.0.0.1:8000/api-token-auth/',
      data: bodyFormData,
      config: { headers: {'Content-Type': 'multipart/form-data' }}
      })
      .then( (response) => {
          //handle success
          this.setState({token:response.data.token,
            token_is_available:true,
            username:'',
            password:'',
          });
          console.log(this.state.token_is_available);
      })
      .catch((response) => {
          //handle error
          console.log(response);
      });

  }
  render() {
    var map = ( <div>
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
</div>);
var form =( <div>
  <MuiThemeProvider>
    <div>
      <Header text='Login'/>
      <Menu />
     <TextField
       hintText="Enter your Username"
       floatingLabelText="Username"
       onChange = {(event,newValue) => this.setState({username:newValue})}
       />
     <br/>
       <TextField
         type="password"
         hintText="Enter your Password"
         floatingLabelText="Password"
         onChange = {(event,newValue) => this.setState({password:newValue})}
         />
       <br/>
       <RaisedButton label="Submit" primary={true} color='red' onClick={this.loginHandler.bind(this)}/>
   </div>
   </MuiThemeProvider>
</div>
)

      if (this.token_is_available){
        return map;

      }
      else {
        return form;
      }
    }
  }



export default geolocated()(MyMap);
