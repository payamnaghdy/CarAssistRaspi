import React, { Component } from 'react';
import ReactDOM from 'react-dom';
import logo from './logo.svg';
import './App.css';
import Header from './Header'
import Sign from './Sign'
import ReactSpeedometer from "react-d3-speedometer"

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
     var presented_sign;
     if(this.state.data[0].Value > 50){
       presented_sign = "1.png"
     }
     else
     {
       presented_sign = ':)';
     }
     ReactDOM.findDOMNode(this.refs.pSign).src = presented_sign
    }
    render() {
          return (
            <div ref="App">
            < Header text="speed"/>
  
            <input value = {this.state.data.id} ref = "myValue"></input>
            <button onClick = {this.setStateHandler.bind(this)}>Update</button>
            <br/>
            <ReactSpeedometer maxValue ={220} value = {this.state.data[0].Value} needleColor="red" startColor="green" endColor="red" segments={10}/>
            <div>
             Warning Legal Speed is {this.state.data[0].Value}
            </div>
            <br/>
            <img ref="pSign"></img>
            </div> 
         );
    }
}

export default Speed;