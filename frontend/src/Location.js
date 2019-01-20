import * as React from 'react';
import { GeolocatedProps, geolocated } from 'react-geolocated';
 
interface IDemoProps {
  label: string;
}
 
class Demo extends React.Component<IDemoProps & GeolocatedProps> {
  render(): JSX.Element {
    return (
      <div>
        label: {this.props.label}
        lattitude: {this.props.coords && this.props.coords.latitude}
        longitude: {this.props.coords && this.props.coords.longitude}
      </div>
    );
  }
}
 
export default geolocated( {
    positionOptions: {
        enableHighAccuracy: true,
        maximumAge: 0,
        timeout: 100,
    },
    watchPosition: true,
    userDecisionTimeout: null,
    suppressLocationOnMount: true,
    geolocationProvider: navigator.geolocation
  })(Demo);