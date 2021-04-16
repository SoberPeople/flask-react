import "@babel/polyfill";
import React from "react";
import ReactDOM from "react-dom";
import { HashRouter, Route } from "react-router-dom";
import First  from "./components/First";
import Second from "./components/Second";

ReactDOM.render(
  <HashRouter>
    <Route exact path="/index/first" component={First} />
    <Route exact path="/index/second" component={Second} />
  </HashRouter>,
  document.getElementById("root")
);
