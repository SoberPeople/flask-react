import React, { Component } from "react";
import { Link, Route } from "react-router-dom";
import DribbleButton from "react-dribble-button";
import "./Home.css";
import Host from "./Host";
import Guest from "./Host";

export default class Home extends Component {
  render() {
    return (
      <div className="button_container">
        <Link to="/host">
          <DribbleButton
            className="button"
            color="cyan"
            onClick={this.onClick}
            animationDuration={1000}
          >
            Host
          </DribbleButton>
        </Link>
        <Link to="/guest">
          <DribbleButton
            className="button"
            color="teal"
            onClick={this.onClick}
            animationDuration={1000}
          >
            Guest
          </DribbleButton>
        </Link>
      </div>
    );
  }
}
