import React, { Component } from 'react';
import ReactDOM from 'react-dom';

class MyButton extends Component{
    clickHandler(){
        
    }
    render() {
        return (
            <button onClick = {this.clickHandler.bind(this)}>{this.props.name}</button>
        );
     }
}