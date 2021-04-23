import "@babel/polyfill";
import React from "react";
import ReactDOM from "react-dom";
import { BrowserRouter, Route } from "react-router-dom";
import { Link } from "react-router-dom";
import Home from "./components/Home";
import Host from "./components/Host";
import Guest from "./components/Guest";

ReactDOM.render(
  <BrowserRouter>
    <Route exact path="/" component={Home} />
    <Route exact path="/host" component={Host} />
    <Route exact path="/guest" component={Guest} />
  </BrowserRouter>,
  document.getElementById("root")
);
