import React, { Component } from 'react';
import ReactDOM from 'react-dom';
import logo from './logo.svg';
import './App.css';
import Header from './Header'
import Sign from './Sign'
import ReactSpeedometer from "react-d3-speedometer"
import Menu from './Menu'
import './speed.css'
class Speed extends Component {
    constructor() {
      super();
      this.state = {
         data: 
         [
            {
               "id":1,
               "Add":"Speed",
               "Value":10
            }
         ]
      }
   }
   setStateHandler() {
     
       this.setState({
       data:
       [
         {
          "id":1,
          "Add":"Speed",
          "Value":ReactDOM.findDOMNode(this.refs.myValue).value
         }
       ]
     })
     //console.log(location.coords.latitude);
     var presented_sign;
     if(this.state.data[0].Value > 50){
       presented_sign = "1.png"
       document.body.style.backgroundColor = "red";
     }
     else
     {
       presented_sign = '1.png';
       document.body.style.backgroundColor = "#F0F0F0";
     }
    }
    render() {
          return (
            <div ref="App">
            < Header text="Speed"/>
            <Menu />
            <input value = {this.state.data.id} ref = "myValue"></input>
            <button onClick = {this.setStateHandler.bind(this)}>Update</button>
            <br/>
            <ReactSpeedometer maxValue ={220} value = {this.state.data[0].Value} needleColor="red" startColor="green" endColor="red" segments={10} width = {window.innerWidth}/>
            
            <br/>
            <div>
             Warning Legal Speed is {this.state.data[0].Value}
            </div>
            </div> 
         );
    }
}

export default Speed;