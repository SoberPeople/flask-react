import React, { Component } from "react";
import ScriptTag from "react-script-tag";

function Guest() {
  return (
    <div>
      <h1>Hello Guest!!!</h1>
      <ScriptTag
        type="text/javascript"
        src="https://unpkg.com/peerjs@1.3.1/dist/peerjs.min.js"
      />
      <ScriptTag src="https://unpkg.com/axios/dist/axios.min.js" />
      <h1>END</h1>
    </div>
  );
}

export default Guest;
