import React, { Component } from 'react';
import ReactDOM from 'react-dom';
import logo from './logo.svg';
import './App.css';
import Header from './Header'
import Sign from './Sign'
class App extends Component {
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
   if(this.state.data[0].Value == 50){
     presented_sign = <Sign/>;
   }
   else
   {
     presented_sign = 'fuck you';
   }
   ReactDOM.findDOMNode(this.refs.pSign).innerHTML = presented_sign;
  }
  render() {
  
    return (
      <div ref="App">
      <Header/>

      <input value = {this.state.data.id} ref = "myValue"></input>
      <button onClick = {this.setStateHandler.bind(this)}>Update</button>
      <br/>
      <div>
      Warning Legal Speed is {this.state.data[0].Value}
      </div>
      <br/>
      <div ref="pSign"></div>
   </div> 
    );
  }
}


export default App;
