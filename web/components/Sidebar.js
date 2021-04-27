import React, { Component } from "react";
import { slide as SideBar } from "react-burger-menu";

export default (props) => {
  return (
    <SideBar {...props}>
      <div> 메세지 창 입니다. </div>
      <div id="msg_input">
        <h6>메세지를 입력하세요</h6>
        <input
          type="text"
          id="sendMessageBox"
          placeholder="Enter a message..."
          autoFocus={true}
        />
        <button type="button" id="sendButton">
          Send
        </button>
        <button type="button" id="clearMsgsButton">
          Clear Msgs (Local)
        </button>
      </div>

      {/* <a className="menu-item" href="/burgers">
        Burgers
      </a>

      <a className="menu-item" href="/pizzas">
        Pizzas
      </a>

      <a className="menu-item" href="/desserts">
        Desserts
      </a> */}
    </SideBar>
  );
};
