import React, { Component } from "react";
import ScriptTag from "react-script-tag";
import "./Guest.css";
import Camera from "./Camera";

function Guest() {
  return (
    <div>
      <ScriptTag
        type="text/javascript"
        src="https://unpkg.com/peerjs@1.3.1/dist/peerjs.min.js"
      />
      <ScriptTag src="https://unpkg.com/axios/dist/axios.min.js" />
      <h1>Hello Guest!!! Welcome to Guest Page</h1>
      <table>
        <thead>
          <tr className="r1">
            <td className="title">Status:</td>
            <td className="title">Student ID:</td>
            <td className="title">Messages:</td>
          </tr>
          <tr className="r2">
            <td>
              <span>ID: </span>
              <input
                type="text"
                id="receiver-id"
                title="Input the ID from receive.html"
                placeholder="Enter Room url..."
              />
            </td>
            <td>
              <input
                type="text"
                id="studentId"
                placeholder="Enter a student id..."
                autoFocus={true}
              />
            </td>
            <td>
              <input
                type="text"
                id="sendMessageBox"
                placeholder="Enter a message..."
                autoFocus={true}
              />
            </td>
          </tr>
        </thead>
      </table>
      <button>Connect</button>
      <h1>-------------VIDEO-------------</h1>
      <Camera />
      <h1>END</h1>
    </div>
  );
}

export default Guest;
