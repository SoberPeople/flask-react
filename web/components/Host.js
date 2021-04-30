import React, { Component } from "react";
import MetaTags from "react-meta-tags";
import ScriptTag from "react-script-tag";

// import Camera from "./Camera";
import Initialize from "../scripts/peer_host";
import SideBar from "./Sidebar";
import "./Sidebar.css";
import Inform from "./Inform";

class Host extends Component {
  render() {
    return (
      <div id="App">
        <ScriptTag
          type="text/javascript"
          src="https://unpkg.com/peerjs@1.3.1/dist/peerjs.min.js"
        />
        <ScriptTag src="https://unpkg.com/axios/dist/axios.min.js" />
        <MetaTags>
          <meta charSet="utf-8" />
          <meta name="viewport" content="width=device-width" />
          <title>Peer-to-Peer Cue System --- Reciever</title>
        </MetaTags>

        <SideBar right pageWrapId={"page-wrap"} outerContainerId={"App"} />

        <h1>Hello Host!!! Welcome to Host Page</h1>
        <div id="page-wrap">
          <table>
            <thead>
              <tr className="r1">
                <Inform className="title" text="Status" />
              </tr>
              <tr className="r2">
                <td>
                  <Inform id="receiver-id" text="ID" />
                </td>
              </tr>
              <tr>
                <td>{/* <Initialize text="커넥션 없음" /> */}</td>
              </tr>
            </thead>
          </table>
          <h1>-------------VIDEO-------------</h1>
          <div className="display-box standby" id="video_grid">
            {/* <Camera /> */}
          </div>
        </div>
      </div>
    );
  }
}
export default Host;
