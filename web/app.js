import "@babel/polyfill";
import React from "react";
import ReactDOM from "react-dom";
import { BrowserRouter, Route } from "react-router-dom";
import First  from "./components/First";
import Second from "./components/Second";

ReactDOM.render(
  <BrowserRouter>
    <Route exact path="/first" component={First} />
    <Route exact path="/second" component={Second} />
  </BrowserRouter>,
  document.getElementById("root")
);
