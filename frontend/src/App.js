import React, { Component } from 'react';
import {BrowserRouter, Route,Switch,Router} from 'react-router-dom'
import ReactDOM from 'react-dom';
import './App.css';
import Header from './Header'
import Speed from './Speed'
import MyMap from './map'
const  newRoute = () => {
  return(
      <div>salam
      </div>
  )
};

class App extends Component { 
  render() {
        return (
            <div>
            <BrowserRouter>
            <Switch>
            <Route path="/speed" component= {Speed} />
            <Route path="/Map" component= {MyMap} />
            </Switch>
            </BrowserRouter>
            </div>
            
        );
    }
  }



export default App;
